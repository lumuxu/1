# -*- coding: utf-8 -*-
"""
batch_train_runs.py

用途：
  依次修改 config.py 里的 FLAGES.iters / log_dir / model_save_dir / data_path，
  然后自动运行 train.py 多次，把每次训练的输出放到不同目录下。

用法（在工程根目录执行）：
  python batch_train_runs.py

可选参数：
  python batch_train_runs.py --iters 1000,10000,20000,50000,500000
  python batch_train_runs.py --runs_dir ./runs_batch
  python batch_train_runs.py --no_restore_config   # 结束后不恢复原 config.py（不推荐）
"""

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ITERS = [1000, 10000, 20000, 50000, 500000]


def _ts():
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _replace_flag(text: str, name: str, value_literal: str) -> str:
    """
    替换 config.py 中 class FLAGES 内形如：
      name=...
      name = ...
    的赋值。只替换第一处匹配。
    """
    # 允许有空格、制表符；允许注释行；尽量只匹配在 class FLAGES 里也行，但简化为全局首个匹配
    pat = re.compile(rf"(^\s*{re.escape(name)}\s*=\s*)(.+?)(\s*(#.*)?$)", re.MULTILINE)
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"在 config.py 里找不到字段：{name}")
    return text[:m.start()] + f"{m.group(1)}{value_literal}{m.group(3)}" + text[m.end():]


def _quote_path(p: Path) -> str:
    # 统一用单引号，避免 Windows 反斜杠转义问题
    return "'" + str(p).replace("\\", "/") + "'"


def _run_one(project_root: Path, train_py: Path, log_path: Path) -> int:
    cmd = [sys.executable, str(train_py)]
    env = os.environ.copy()
    # 如果你想每次固定显卡/线程，这里也可加 env 设置
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[INFO] cmd: {' '.join(cmd)}\n")
        f.write(f"[INFO] cwd: {project_root}\n\n")
        f.flush()
        p = subprocess.run(
            cmd,
            cwd=str(project_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(p.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=str, default=",".join(map(str, DEFAULT_ITERS)),
                    help="逗号分隔的迭代次数列表，例如 1000,10000,20000")
    ap.add_argument("--project_root", type=str, default=".",
                    help="工程根目录（默认当前目录）")
    ap.add_argument("--config", type=str, default="config.py",
                    help="config.py 路径（相对 project_root 或绝对路径）")
    ap.add_argument("--train", type=str, default="train.py",
                    help="train.py 路径（相对 project_root 或绝对路径）")
    ap.add_argument("--runs_dir", type=str, default="./runs_batch",
                    help="批量运行输出目录（相对 project_root 或绝对路径）")
    ap.add_argument("--no_restore_config", action="store_true",
                    help="结束后不恢复原 config.py（不推荐）")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = (Path(args.config) if Path(args.config).is_absolute() else (project_root / args.config)).resolve()
    train_py = (Path(args.train) if Path(args.train).is_absolute() else (project_root / args.train)).resolve()
    runs_dir = (Path(args.runs_dir) if Path(args.runs_dir).is_absolute() else (project_root / args.runs_dir)).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config.py：{config_path}")
    if not train_py.exists():
        raise FileNotFoundError(f"找不到 train.py：{train_py}")

    iters_list = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
    if not iters_list:
        raise ValueError("iters 列表为空")

    _ensure_dir(runs_dir)

    # 备份 config.py
    backup_path = config_path.with_suffix(config_path.suffix + f".bak_batch_{_ts()}")
    shutil.copy2(str(config_path), str(backup_path))
    print(f"[INFO] 已备份 config.py -> {backup_path}")

    original_text = _read_text(config_path)

    try:
        for iters in iters_list:
            run_name = f"iters_{iters}"
            run_dir = runs_dir / run_name
            # 如果已存在，避免覆盖
            if run_dir.exists():
                run_dir = runs_dir / f"{run_name}_{_ts()}"
            _ensure_dir(run_dir)

            log_dir = run_dir / "log"
            model_dir = run_dir / "model"
            cache_dir = run_dir / "cache"
            _ensure_dir(log_dir)
            _ensure_dir(model_dir)
            _ensure_dir(cache_dir)

            # 对每次运行，给一个独立的 h5 缓存，避免互相覆盖
            data_path = cache_dir / "train_cache.h5"

            # 修改 config.py
            text = original_text
            text = _replace_flag(text, "iters", str(iters))
            text = _replace_flag(text, "log_dir", _quote_path(log_dir))
            text = _replace_flag(text, "model_save_dir", _quote_path(model_dir))
            text = _replace_flag(text, "data_path", _quote_path(data_path))
            _write_text(config_path, text)

            # 保存一份本次使用的 config 快照
            _write_text(run_dir / "config_used.py", text)

            # 运行 train.py
            out_log = run_dir / "stdout_stderr.log"
            print(f"[INFO] 开始训练：iters={iters}  输出目录：{run_dir}")
            rc = _run_one(project_root, train_py, out_log)
            print(f"[INFO] 结束训练：iters={iters}  returncode={rc}  日志：{out_log}")

            # 如果非 0，继续下一次还是停止？这里默认停止，避免静默失败
            if rc != 0:
                print("[ERROR] train.py 运行失败，已停止后续批量训练。请查看日志定位问题：", out_log)
                break

    except KeyboardInterrupt:
        print("\n[WARN] 收到 Ctrl+C，中断批量训练。")
    finally:
        if args.no_restore_config:
            print("[WARN] --no_restore_config 已启用：不会恢复原 config.py")
        else:
            _write_text(config_path, original_text)
            print(f"[INFO] 已恢复原 config.py：{config_path}")
            print(f"[INFO] 原始备份仍保留在：{backup_path}")

    print("[DONE] 批量训练脚本结束。输出目录：", runs_dir)


if __name__ == "__main__":
    main()

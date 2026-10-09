"""`python -m harness` → 统一入口（见 harness/run_cli.py）。

用 __main__.py 而不是 run.py：后者已是「单次 run 的落盘逻辑」，
名字被占；而 `python -m harness` 比 `python -m harness.run_cli` 更短，
也更接近 `harbor run` 的手感。
"""
from .run_cli import main

if __name__ == "__main__":
    main()

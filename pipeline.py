"""命令行兼容入口：完整流程实现在 astock_core.pipeline。"""

from astock_core.pipeline import *  # noqa: F401,F403
from astock_core.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())

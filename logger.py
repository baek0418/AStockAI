"""AStockAI 统一日志模块。"""

import logging
from datetime import datetime
from pathlib import Path


def get_logger(name):
    """获取同时输出到终端和日志文件的统一日志对象。"""
    logger = logging.getLogger(name)

    if getattr(logger, "_astockai_configured", False):
        return logger

    project_directory = Path(__file__).parent
    logs_directory = project_directory / "logs"
    logs_directory.mkdir(exist_ok=True)
    log_file = logs_directory / f"astockai_{datetime.now().strftime('%Y-%m-%d')}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger._astockai_configured = True

    return logger

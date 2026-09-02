"""面向行情更新和研究训练的结构化过程日志。"""

import json
import threading
from datetime import datetime
from pathlib import Path

from astock_core.runtime.logger import get_logger


class ProcessJournal:
    """同时写入人类可读日志与 JSONL 事件日志，不记录令牌等敏感配置。"""

    def __init__(self, process_name, project_directory=None):
        self.process_name = str(process_name)
        project = Path(project_directory or Path(__file__).parents[2])
        self.logs_directory = project / "logs"
        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.event_file = self.logs_directory / f"{self.process_name}_{datetime.now():%Y-%m-%d}.jsonl"
        self.logger = get_logger(f"astock_core.{self.process_name}")
        self._lock = threading.Lock()

    def event(self, stage, status="info", **details):
        record = {
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "流程": self.process_name,
            "阶段": str(stage),
            "状态": str(status),
            **details,
        }
        message = json.dumps(record, ensure_ascii=False, sort_keys=True)
        log_method = self.logger.error if status in {"failed", "error"} else self.logger.warning if status == "partial" else self.logger.info
        log_method(message)
        with self._lock:
            with self.event_file.open("a", encoding="utf-8") as file:
                file.write(message + "\n")
        return record

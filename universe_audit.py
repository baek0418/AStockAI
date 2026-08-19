"""研究股票池的数据覆盖与历史质量审计。"""

import json
from datetime import datetime
from pathlib import Path

from prediction_features import load_history_csv
from stock_universe import get_enabled_stock_universe


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
MIN_HISTORY_DAYS = 480


def audit_research_universe(project_directory=PROJECT_DIRECTORY):
    root = Path(project_directory)
    records = []
    for stock in get_enabled_stock_universe():
        file_path = root / "data" / f"{stock['name']}历史.csv"
        record = {"代码": stock["code"], "名称": stock["name"], "来源": stock["source"], "状态": "未下载"}
        try:
            history = load_history_csv(file_path)
            record.update({"日线数量": len(history), "最新日期": history.iloc[-1]["日期"].date().isoformat()})
            record["状态"] = "可研究" if len(history) >= MIN_HISTORY_DAYS else "历史不足"
        except (OSError, ValueError):
            pass
        records.append(record)
    ready = [item for item in records if item["状态"] == "可研究"]
    latest_dates = [item["最新日期"] for item in ready if item.get("最新日期")]
    return {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "最少历史日线": MIN_HISTORY_DAYS,
        "股票总数": len(records),
        "可研究数量": len(ready),
        "覆盖率": round(len(ready) / len(records) * 100, 2) if records else 0,
        "最新可用日期": min(latest_dates) if latest_dates else None,
        "股票": records,
    }


def save_audit(report, output_directory=PROJECT_DIRECTORY / "output"):
    path = Path(output_directory) / f"research_universe_audit_{datetime.now():%Y-%m-%d}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    report = audit_research_universe()
    path = save_audit(report)
    print(f"研究股票池：{report['可研究数量']}/{report['股票总数']} 可研究，覆盖率 {report['覆盖率']}%")
    print(path)

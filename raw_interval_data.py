"""构建未来收益区间实验专用的原始日线快照，不改写正式前复权数据。"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from market_data_sources import RequestPacer, RetryPolicy, fetch_raw_daily_history
from process_journal import ProcessJournal
from stock_universe import get_enabled_stock_universe


def _save_raw_history(stock, data_directory, fetch_history):
    result = fetch_history(stock["market_code"])
    data_file = Path(data_directory) / f"{stock['name']}历史.csv"
    temporary_file = data_file.with_suffix(".csv.tmp")
    result.history.to_csv(temporary_file, index=False, encoding="utf-8-sig")
    temporary_file.replace(data_file)
    provenance_directory = Path(data_directory) / "provenance"
    provenance_directory.mkdir(parents=True, exist_ok=True)
    provenance_file = provenance_directory / f"{stock['code']}.json"
    temporary_provenance = provenance_file.with_suffix(".json.tmp")
    provenance = result.provenance(stock["code"], stock["market_code"])
    provenance["用途"] = "未来5日收益区间研究专用原始价格快照；不参与正式量化、日报或持仓估值。"
    temporary_provenance.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_provenance.replace(provenance_file)
    return {
        "code": stock["code"], "name": stock["name"], "source": result.source,
        "adjustment": result.adjustment, "rows": int(len(result.history)),
        "used_fallback": result.used_fallback,
    }


def snapshot_raw_interval_data(
    project_directory=None,
    stock_universe=None,
    max_workers=3,
    request_interval_seconds=0.35,
    max_attempts=3,
    limit=None,
    fetch_history=fetch_raw_daily_history,
):
    """显式下载原始价格快照；失败保留既有文件，返回可审计的批处理结果。"""
    project_directory = Path(project_directory or Path(__file__).parent)
    data_directory = project_directory / "data" / "raw_interval"
    data_directory.mkdir(parents=True, exist_ok=True)
    journal = ProcessJournal("raw_interval_data", project_directory)
    stocks = list(stock_universe) if stock_universe is not None else list(get_enabled_stock_universe())
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit 必须是正整数。")
        stocks = stocks[:limit]
    if not stocks:
        return {"status": "failed", "message": "股票池为空。", "details": {}}
    pacer = RequestPacer(request_interval_seconds)
    retry_policy = RetryPolicy(max_attempts=max_attempts, initial_backoff_seconds=0.5)

    def fetch(market_code):
        return fetch_history(market_code, retry_policy=retry_policy, pacer=pacer)

    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="原始日线") as executor:
        futures = {executor.submit(_save_raw_history, stock, data_directory, fetch): stock for stock in stocks}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append({"code": stock["code"], "name": stock["name"], "error": str(error)})
    successes.sort(key=lambda item: item["code"])
    failures.sort(key=lambda item: item["code"])
    status = "success" if not failures else "partial" if successes else "failed"
    details = {
        "success_count": len(successes), "failed_count": len(failures),
        "fallback_count": sum(item["used_fallback"] for item in successes),
        "successes": successes, "failures": failures, "data_directory": str(data_directory),
    }
    journal.event("完成原始价格快照", status, 成功股票数=len(successes), 失败股票数=len(failures))
    if status == "success":
        message = "原始价格快照完成。"
    elif status == "partial":
        message = "原始价格快照部分完成。"
    else:
        message = "原始价格快照失败；既有快照未被覆盖。"
    return {"status": status, "message": message, "details": details}


def main():
    parser = argparse.ArgumentParser(description="构建未来收益区间实验专用原始日线快照。")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, help="仅下载前 N 只股票，用于诊断原始行情链路。")
    arguments = parser.parse_args()
    result = snapshot_raw_interval_data(max_workers=arguments.max_workers, limit=arguments.limit)
    print(result["message"])
    print(json.dumps({key: result["details"].get(key) for key in ("success_count", "failed_count", "fallback_count")}, ensure_ascii=False))
    return 0 if result["status"] in {"success", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

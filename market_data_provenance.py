"""核验本地历史 CSV 与其腾讯行情来源的复权方式。

只读取本地 CSV 并请求同一公开日线接口进行比对；不会更新或覆盖任何历史文件。
未能明确匹配为前复权或未复权的数据一律标为 ``unverified``，不能进入 Qlib 研究。
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from market_data_adapter import REQUIRED_SOURCE_COLUMNS, read_history_csv
from stock_universe import create_market_code
from update_data import create_history_dataframe


FIELDS = ["开盘", "收盘", "最高", "最低", "成交量"]
QUERY_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def fetch_adjustment_candidates(market_code, request_get=requests.get):
    """拉取同一请求内的前复权与未复权候选日线。"""
    response = request_get(
        QUERY_URL,
        params={"param": f"{market_code},day,,,600,qfq"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    try:
        payload = response.json()["data"][market_code]
    except (KeyError, TypeError) as error:
        raise ValueError(f"行情接口未返回 {market_code} 的数据。") from error
    if not isinstance(payload, dict):
        raise ValueError(f"行情接口返回的 {market_code} 数据格式无效。")
    candidates = {}
    for adjustment, payload_key in (("qfq", "qfqday"), ("none", "day")):
        rows = payload.get(payload_key)
        if isinstance(rows, list) and rows:
            candidates[adjustment] = create_history_dataframe(rows)
    if not candidates:
        raise ValueError(f"行情接口未返回 {market_code} 的有效 qfqday 或 day 数据。")
    return candidates


def compare_history(local_history, candidate_history):
    """比较日期交集上的完整 OHLCV；返回重叠数、匹配数与匹配率。"""
    local = local_history[["日期", *FIELDS]].copy().set_index("日期").sort_index()
    candidate = candidate_history.copy()
    candidate["日期"] = pd.to_datetime(candidate["日期"], errors="coerce")
    for field in FIELDS:
        candidate[field] = pd.to_numeric(candidate[field], errors="coerce")
    candidate = candidate.dropna(subset=["日期", *FIELDS]).drop_duplicates("日期", keep="last")
    candidate = candidate[["日期", *FIELDS]].set_index("日期").sort_index()
    merged = local.join(candidate, how="inner", lsuffix="_local", rsuffix="_candidate")
    if merged.empty:
        return {"overlap_rows": 0, "matched_rows": 0, "match_rate": 0.0}
    matches = pd.Series(True, index=merged.index)
    for field in FIELDS:
        matches &= (merged[f"{field}_local"] - merged[f"{field}_candidate"]).abs() <= 1e-8
    mismatch_dates = merged.index[~matches].strftime("%Y-%m-%d").tolist()
    return {
        "overlap_rows": int(len(merged)),
        "matched_rows": int(matches.sum()),
        "match_rate": round(float(matches.mean()), 8),
        "mismatch_dates": mismatch_dates,
    }


def classify_adjustment(local_history, candidates):
    """只有唯一候选在充分重叠的每行 OHLCV 都相等时才确认复权方式。"""
    comparisons = {name: compare_history(local_history, frame) for name, frame in candidates.items()}
    local_rows = len(local_history)
    exact = [
        name
        for name, result in comparisons.items()
        if result["overlap_rows"] >= min(300, local_rows) and result["match_rate"] == 1.0
    ]
    refresh_drift = [
        name
        for name, result in comparisons.items()
        if result["overlap_rows"] >= min(300, local_rows)
        and result["match_rate"] >= 0.995
        and result["mismatch_dates"] == [local_history["日期"].max().strftime("%Y-%m-%d")]
    ]
    valid = exact or refresh_drift
    if len(valid) == 1:
        status = "verified"
        adjustment = valid[0]
    elif len(valid) > 1:
        status = "ambiguous"
        adjustment = "unverified"
    else:
        status = "unmatched"
        adjustment = "unverified"
    return {"status": status, "adjustment": adjustment, "comparisons": comparisons}


def audit_local_adjustments(data_directory, code_lookup, fetch_candidates=fetch_adjustment_candidates):
    """仅扫描 data 根目录的 OHLCV 文件，逐一给出可追溯的复权核验结果。"""
    results = []
    for history_file in sorted(Path(data_directory).glob("*.csv")):
        stock_name = history_file.stem.replace("历史", "")
        try:
            columns = set(pd.read_csv(history_file, nrows=0, encoding="utf-8-sig").columns)
            if not REQUIRED_SOURCE_COLUMNS.issubset(columns):
                continue
            stock_code = code_lookup.get(stock_name)
            if not stock_code:
                raise ValueError("股票代码未映射，无法核验来源。")
            local_history, _ = read_history_csv(history_file)
            result = classify_adjustment(local_history, fetch_candidates(create_market_code(stock_code)))
            results.append(
                {
                    "file": history_file.name,
                    "stock_name": stock_name,
                    "stock_code": stock_code,
                    "market_code": create_market_code(stock_code),
                    "local_rows": int(len(local_history)),
                    **result,
                }
            )
        except (OSError, UnicodeDecodeError, ValueError, requests.RequestException, pd.errors.ParserError) as error:
            results.append({"file": history_file.name, "stock_name": stock_name, "status": "failed", "reason": str(error)})
    return results


def run_adjustment_audit(project_directory=None):
    """保存可供 Qlib 导出层读取的复权来源清单。"""
    project = Path(project_directory or Path(__file__).parent)
    from market_data_adapter import load_stock_code_lookup

    results = audit_local_adjustments(project / "data", load_stock_code_lookup(project))
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "只读 data/*.csv；使用腾讯行情接口 qfqday/day 与本地 OHLCV 按日期比对。",
        "verified_qfq": sum(item.get("adjustment") == "qfq" for item in results),
        "verified_none": sum(item.get("adjustment") == "none" for item in results),
        "unverified": sum(item.get("adjustment") == "unverified" for item in results),
        "failed": sum(item.get("status") == "failed" for item in results),
        "files": results,
    }
    output_directory = project / "output" / "research"
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"market_data_adjustment_audit_{datetime.now():%Y-%m-%d}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, path


if __name__ == "__main__":
    summary, output_file = run_adjustment_audit()
    print(output_file)
    print({key: summary[key] for key in ("verified_qfq", "verified_none", "unverified", "failed")})

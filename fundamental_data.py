"""A 股基本面事实快照：显式下载、原始字段留存、报告阶段只读。"""

import argparse
import json
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import requests

from stock_universe import normalize_stock_code


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
FUNDAMENTAL_DIRECTORY = PROJECT_DIRECTORY / "data" / "fundamentals"
EASTMONEY_FINANCE_URL = "https://datacenter.eastmoney.com/securities/api/data/get"
EASTMONEY_PROFILE_URL = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
SOURCE_NAME = "东方财富公开财务指标接口（聚合数据，须以官方定期报告复核）"
CNINFO_STOCK_PAGE = "https://www.cninfo.com.cn/new/disclosure/stock?stockCode={code}"
METRICS = {
    "营业总收入": "TOTALOPERATEREVE",
    "归母净利润": "PARENTNETPROFIT",
    "营业总收入同比增长": "TOTALOPERATEREVETZ",
    "归母净利润同比增长": "PARENTNETPROFITTZ",
    "扣非净利润同比增长": "KCFJCXSYJLRTZ",
    "净资产收益率(加权)": "ROEJQ",
    "销售毛利率": "XSMLL",
    "资产负债率": "ZCFZL",
    "每股经营现金流": "MGJYXJJE",
    "每股收益(基本)": "EPSJB",
    "每股净资产": "BPS",
}
PROFILE_FIELDS = {
    "所属行业": "EM2016",
    "证监会行业": "INDUSTRYCSRC1",
    "主营业务": "MAIN_BUSINESS",
    "上市市场": "TRADE_MARKET",
    "实际控制人": "ACTUAL_HOLDER",
    "上市日期": "LISTING_DATE",
}
PEER_METRICS = {
    "营业总收入同比增长": ("TOTALOPERATEREVETZ", "higher"),
    "归母净利润同比增长": ("PARENTNETPROFITTZ", "higher"),
    "净资产收益率(加权)": ("ROEJQ", "higher"),
    "资产负债率": ("ZCFZL", "lower"),
}


def _security_code(code):
    normalized = normalize_stock_code(code)
    return f"{normalized}.SH" if normalized.startswith("6") else f"{normalized}.SZ"


def get_snapshot_file(stock_code, directory=FUNDAMENTAL_DIRECTORY):
    return Path(directory) / f"{normalize_stock_code(stock_code)}_fundamentals.json"


def _request_parameters(stock_code):
    return {
        "type": "RPT_F10_FINANCE_MAINFINADATA",
        "sty": "APP_F10_MAINFINADATA",
        "quoteColumns": "",
        "filter": f'(SECUCODE="{_security_code(stock_code)}")',
        "p": 1,
        "ps": 12,
        "sr": -1,
        "st": "REPORT_DATE",
        "source": "HSF10",
        "client": "PC",
    }


def _request_headers():
    return {
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://emweb.securities.eastmoney.com/",
        "User-Agent": "Mozilla/5.0",
    }


def _profile_request_parameters(stock_code):
    return {
        "reportName": "RPT_F10_BASIC_ORGINFO",
        "columns": "ALL",
        "quoteColumns": "",
        "filter": f'(SECUCODE="{_security_code(stock_code)}")',
        "pageNumber": 1,
        "pageSize": 1,
        "sortTypes": "",
        "sortColumns": "",
        "source": "HSF10",
        "client": "PC",
    }


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _extract_records(payload):
    records = (payload or {}).get("result", {}).get("data", [])
    if not isinstance(records, list) or not records or not all(isinstance(item, dict) for item in records):
        raise ValueError("财务指标接口未返回有效报告记录。")
    return records


def build_fundamental_snapshot(stock_code, records, fetched_at=None):
    """规范化接口记录；保留近期原始行，方便之后核验字段含义和报告期。"""
    code = normalize_stock_code(stock_code)
    records = _extract_records({"result": {"data": records}})
    records = sorted(
        records,
        key=lambda item: str(item.get("REPORT_DATE") or item.get("NOTICE_DATE") or ""),
        reverse=True,
    )
    latest = records[0]
    returned_code = str(latest.get("SECURITY_CODE") or latest.get("SECUCODE") or "").strip()[:6]
    if returned_code and returned_code != code:
        raise ValueError(f"财务指标接口返回代码 {returned_code}，与请求代码 {code} 不一致。")
    metrics = {label: latest.get(field) for label, field in METRICS.items()}
    return {
        "数据状态": "可用",
        "股票代码": code,
        "证券代码": latest.get("SECUCODE", _security_code(code)),
        "股票名称": latest.get("SECURITY_NAME_ABBR", "未提供"),
        "来源": SOURCE_NAME,
        "官方核验页": CNINFO_STOCK_PAGE.format(code=code),
        "下载时间": fetched_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "最新报告": {
            "报告期": latest.get("REPORT_DATE") or latest.get("REPORT_DATE_NAME") or "未提供",
            "报告类型": latest.get("REPORT_TYPE") or latest.get("REPORT_DATE_NAME") or "未提供",
            "公告日期": latest.get("NOTICE_DATE") or "未提供",
            "货币单位": latest.get("CURRENCY") or "未提供",
            "指标": metrics,
        },
        "原始近期报告": records[:4],
        "说明": "收入、利润等绝对值的单位以来源字段为准；同比字段按来源原样保留。报告必须结合巨潮资讯官方定期报告复核。",
    }


def build_company_profile(record):
    """规范化公司和行业字段；只保留资料源直接提供的描述。"""
    if not isinstance(record, dict):
        return {"数据状态": "数据不足：公司与行业画像未返回有效记录。"}
    return {
        "数据状态": "可用",
        "来源": "东方财富公开公司资料接口（聚合数据，须以官方披露复核）",
        "字段": {label: record.get(field) for label, field in PROFILE_FIELDS.items()},
        "原始记录": record,
    }


def _write_atomically(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def collect_fundamental_snapshot(stock_code, directory=FUNDAMENTAL_DIRECTORY, request_get=requests.get):
    """显式下载单股财务指标；请求或校验失败时不创建、不覆盖快照。"""
    code = normalize_stock_code(stock_code)
    try:
        response = request_get(
            EASTMONEY_FINANCE_URL,
            params=_request_parameters(code),
            headers=_request_headers(),
            timeout=15,
        )
        response.raise_for_status()
        snapshot = build_fundamental_snapshot(code, _extract_records(response.json()))
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        return {"status": "failed", "message": f"下载基本面快照失败：{error}。已有本地快照未被改动。"}
    try:
        profile_response = request_get(
            EASTMONEY_PROFILE_URL,
            params=_profile_request_parameters(code),
            headers=_request_headers(),
            timeout=15,
        )
        profile_response.raise_for_status()
        profile_records = _extract_records(profile_response.json())
        snapshot["公司与行业画像"] = build_company_profile(profile_records[0])
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        snapshot["公司与行业画像"] = {"数据状态": f"数据不足：公司与行业画像下载失败：{error}。"}
    output_file = _write_atomically(get_snapshot_file(code, directory), snapshot)
    return {
        "status": "success",
        "message": "基本面快照已下载；请结合巨潮资讯官方定期报告复核。",
        "file": str(output_file),
        "snapshot": snapshot,
    }


def load_fundamental_snapshot(stock_code, directory=FUNDAMENTAL_DIRECTORY):
    """读取已经下载的基本面快照，不联网。"""
    snapshot_file = get_snapshot_file(stock_code, directory)
    if not snapshot_file.is_file():
        return {"数据状态": "数据不足：未下载基本面快照。"}
    try:
        snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"数据状态": f"数据不足：基本面快照读取失败：{error}。"}
    if not isinstance(snapshot, dict) or snapshot.get("数据状态") != "可用":
        return {"数据状态": "数据不足：基本面快照格式无效。"}
    return snapshot


def summarize_fundamental_evidence(snapshot):
    """只呈现已下载报告的事实，不将不同报告类型自行计算为增速。"""
    if not isinstance(snapshot, dict) or snapshot.get("数据状态") != "可用":
        return {"数据状态": (snapshot or {}).get("数据状态", "数据不足：未下载基本面快照。"), "事实": []}
    report = snapshot.get("最新报告", {})
    metrics = report.get("指标", {})
    facts = [
        f"最新报告期：{report.get('报告期', '未提供')}；公告日期：{report.get('公告日期', '未提供')}。",
    ]
    for label in ("营业总收入", "归母净利润", "营业总收入同比增长", "归母净利润同比增长", "净资产收益率(加权)", "资产负债率", "每股经营现金流"):
        value = metrics.get(label)
        if value is not None:
            facts.append(f"{label}：{value}（按来源原始口径）。")
    profile = snapshot.get("公司与行业画像", {})
    profile_fields = profile.get("字段", {}) if profile.get("数据状态") == "可用" else {}
    for label in ("所属行业", "证监会行业", "主营业务", "上市市场", "实际控制人"):
        value = profile_fields.get(label)
        if value is not None and str(value).strip():
            text = str(value).strip()
            facts.append(f"{label}：{text[:240]}（按来源原始描述）。")
    return {
        "数据状态": "可用",
        "事实": facts,
        "来源": snapshot.get("来源"),
        "官方核验页": snapshot.get("官方核验页"),
        "报告期": report.get("报告期"),
        "公告日期": report.get("公告日期"),
        "指标": metrics,
        "公司与行业画像状态": profile.get("数据状态", "数据不足：未下载公司与行业画像。"),
    }


def build_valuation_observation(fundamental_evidence, latest_close):
    """以最新本地收盘与已披露每股指标做口径透明的估值观察，不外推为合理价值。"""
    close = _number(latest_close)
    metrics = (fundamental_evidence or {}).get("指标", {})
    if close is None or close <= 0 or not metrics:
        return {"数据状态": "数据不足：缺少最新本地收盘或每股财务指标。"}
    result = {
        "数据状态": "可用",
        "价格日期": fundamental_evidence.get("价格日期", "本地价格证据日期未提供"),
        "报告期": fundamental_evidence.get("报告期", "未提供"),
        "最新收盘": round(close, 4),
        "市净率(PB)": None,
        "静态市盈率(PE)": None,
        "说明": "PB=最新本地收盘/最新报告每股净资产；静态PE只在最近报告为年报时展示，不是预测估值。",
    }
    bps = _number(metrics.get("每股净资产"))
    if bps is not None and bps > 0:
        result["市净率(PB)"] = round(close / bps, 2)
    eps = _number(metrics.get("每股收益(基本)"))
    report_date = str(result["报告期"])
    if eps is not None and eps > 0 and report_date.startswith(("20", "19")) and report_date[5:10] == "12-31":
        result["静态市盈率(PE)"] = round(close / eps, 2)
    return result


def _load_snapshot_file(snapshot_file):
    try:
        data = json.loads(Path(snapshot_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("数据状态") == "可用" else None


def build_industry_peer_comparison(snapshot, directory=FUNDAMENTAL_DIRECTORY, minimum_peers=3):
    """在同一行业、同一报告期的已下载快照中做横向事实比较。

    不自动下载同行数据，也不把不同报告期或不同会计口径混在同一排名中。
    """
    profile = (snapshot or {}).get("公司与行业画像", {})
    industry = (profile.get("字段", {}) if profile.get("数据状态") == "可用" else {}).get("所属行业")
    report_period = (snapshot or {}).get("最新报告", {}).get("报告期")
    target_code = str((snapshot or {}).get("股票代码", ""))
    if not industry or not report_period or not target_code:
        return {"数据状态": "数据不足：目标公司缺少行业或报告期，不能进行同业比较。"}
    peers = []
    for snapshot_file in Path(directory).glob("*_fundamentals.json"):
        candidate = _load_snapshot_file(snapshot_file)
        candidate_profile = (candidate or {}).get("公司与行业画像", {})
        candidate_industry = (
            candidate_profile.get("字段", {}) if candidate_profile.get("数据状态") == "可用" else {}
        ).get("所属行业")
        candidate_period = (candidate or {}).get("最新报告", {}).get("报告期")
        if candidate_industry != industry or candidate_period != report_period:
            continue
        peers.append(candidate)
    if len(peers) < minimum_peers:
        return {
            "数据状态": f"数据不足：行业“{industry}”在报告期 {report_period} 仅有 {len(peers)} 家已下载快照，至少需要 {minimum_peers} 家。",
            "所属行业": industry,
            "报告期": report_period,
            "可比公司数量": len(peers),
        }

    comparison = {}
    for label, (field, direction) in PEER_METRICS.items():
        values = []
        for peer in peers:
            value = _number(peer.get("最新报告", {}).get("指标", {}).get(label))
            if value is not None:
                values.append((str(peer.get("股票代码", "")), value))
        target_value = next((value for code, value in values if code == target_code), None)
        if target_value is None or len(values) < minimum_peers:
            comparison[label] = {"数据状态": "数据不足：有效同业指标不足。"}
            continue
        ranked = sorted(values, key=lambda item: (item[1], item[0]), reverse=(direction == "higher"))
        rank = next(index for index, (code, _) in enumerate(ranked, start=1) if code == target_code)
        comparison[label] = {
            "数据状态": "可用",
            "本公司": round(target_value, 4),
            "同业中位数": round(float(statistics.median(value for _, value in values)), 4),
            "同业排名": rank,
            "有效可比公司数": len(values),
            "方向": "数值较高通常更有利" if direction == "higher" else "数值较低通常更稳健（行业口径仍需结合业务模式）",
        }
    return {
        "数据状态": "可用",
        "所属行业": industry,
        "报告期": report_period,
        "可比公司数量": len(peers),
        "指标比较": comparison,
        "说明": "仅比较本地已下载、所属行业完全一致且报告期相同的快照；不是全行业覆盖，也不构成行业评级。",
    }


def _enabled_watchlist_codes(watchlist_file):
    try:
        data = json.loads(Path(watchlist_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取关注列表：{error}。") from error
    stocks = data.get("stocks", []) if isinstance(data, dict) else []
    if not isinstance(stocks, list):
        raise ValueError("关注列表中的 stocks 必须是列表。")
    codes = []
    for item in stocks:
        if not isinstance(item, dict) or item.get("enable", True) is False:
            continue
        try:
            codes.append(normalize_stock_code(item.get("code", "")))
        except ValueError:
            continue
    return sorted(set(codes))


def _is_snapshot_stale(snapshot, now, max_age_days):
    if not isinstance(snapshot, dict) or snapshot.get("数据状态") != "可用":
        return True
    try:
        fetched_at = datetime.strptime(str(snapshot.get("下载时间", "")), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return True
    return now - fetched_at > timedelta(days=max_age_days)


def refresh_watchlist_fundamentals(
    watchlist_file=PROJECT_DIRECTORY / "watchlist.json",
    directory=FUNDAMENTAL_DIRECTORY,
    max_age_days=7,
    force=False,
    request_get=requests.get,
    now=None,
):
    """刷新启用关注股中过期的基本面快照；单股失败不影响其他股票。"""
    if max_age_days < 0:
        raise ValueError("max_age_days 不能为负数。")
    codes = _enabled_watchlist_codes(watchlist_file)
    current_time = now or datetime.now()
    refreshed, skipped, failed = [], [], []
    for code in codes:
        snapshot = load_fundamental_snapshot(code, directory)
        if not force and not _is_snapshot_stale(snapshot, current_time, max_age_days):
            skipped.append(code)
            continue
        result = collect_fundamental_snapshot(code, directory, request_get=request_get)
        if result["status"] == "success":
            refreshed.append(code)
        else:
            failed.append({"code": code, "message": result["message"]})
    return {
        "status": "success" if not failed else "partial",
        "关注股数量": len(codes),
        "已更新": refreshed,
        "仍有效无需更新": skipped,
        "失败": failed,
        "说明": "仅刷新缺失或超过时效的启用关注股快照；失败不会删除已有快照。",
    }


def main():
    parser = argparse.ArgumentParser(description="显式下载单股基本面研究快照")
    parser.add_argument("stock_code", nargs="?", help="六位沪深 A 股代码")
    parser.add_argument("--directory", default=str(FUNDAMENTAL_DIRECTORY))
    parser.add_argument("--watchlist", action="store_true", help="刷新启用关注股中过期的基本面快照。")
    parser.add_argument("--watchlist-file", default=str(PROJECT_DIRECTORY / "watchlist.json"))
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument("--force", action="store_true", help="刷新时忽略快照时效。")
    args = parser.parse_args()
    if args.watchlist:
        try:
            result = refresh_watchlist_fundamentals(
                args.watchlist_file, args.directory, args.max_age_days, args.force
            )
        except ValueError as error:
            print(f"基本面批量刷新失败：{error}")
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"success", "partial"} else 1
    if not args.stock_code:
        parser.error("请提供股票代码，或使用 --watchlist。")
    result = collect_fundamental_snapshot(args.stock_code, args.directory)
    print(result["message"])
    if result.get("file"):
        print(result["file"])
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

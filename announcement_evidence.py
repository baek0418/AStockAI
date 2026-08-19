"""巨潮资讯公告事实快照：只收集可追溯标题，不生成公告内容解读。"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
SOURCE_NAME = "巨潮资讯（官方披露）"


def _load_enabled_stocks(watchlist_file):
    try:
        data = json.loads(Path(watchlist_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    stocks = data.get("stocks", []) if isinstance(data, dict) else []
    return [
        item for item in stocks
        if isinstance(item, dict) and item.get("enable", True) is not False
        and str(item.get("code", "")).strip()
    ]


def classify_announcement_title(title):
    """仅按标题做中性归类，不推断事件影响。"""
    title = str(title or "")
    categories = (
        ("定期披露", ("年度报告", "半年度报告", "季度报告", "业绩预告", "业绩快报")),
        ("公司治理", ("董事会", "监事会", "股东大会")),
        ("权益与融资", ("回购", "增持", "减持", "发行", "可转债", "股权激励")),
        ("重大事项", ("重组", "收购", "合同", "中标", "诉讼", "仲裁", "担保")),
    )
    for category, keywords in categories:
        if any(keyword in title for keyword in keywords):
            return category
    return "其他披露"


def _create_client():
    try:
        from easy_tdx.cninfo import CninfoClient
    except ImportError as error:
        raise RuntimeError("公告数据依赖未安装：请安装 easy-tdx。") from error
    return CninfoClient(timeout=10)


def collect_announcement_snapshot(watchlist_file, output_directory, report_date=None, client=None, count=5):
    """为启用关注股建立当日官方公告快照；单股失败不阻断其余股票。"""
    output_directory = Path(output_directory)
    report_date = report_date or datetime.now().strftime("%Y-%m-%d")
    snapshot_file = output_directory / f"announcement_snapshot_{report_date}.json"
    if snapshot_file.is_file():
        return snapshot_file

    client = client or _create_client()
    records = []
    for stock in _load_enabled_stocks(watchlist_file):
        code = str(stock["code"]).strip().zfill(6)
        name = str(stock.get("name", "未知股票")).strip() or "未知股票"
        try:
            frame = client.get_announcements(code, count=count)
        except Exception as error:  # 外部源失败需要可审计但不暴露内部配置。
            records.append({"股票代码": code, "股票名称": name, "数据状态": "获取失败", "公告": [], "说明": str(error)})
            continue
        announcements = []
        if isinstance(frame, pd.DataFrame):
            for _, item in frame.head(count).iterrows():
                title = str(item.get("title", "") or "")
                announcements.append({
                    "日期": str(item.get("date", "") or ""),
                    "标题": title,
                    "类别": classify_announcement_title(title),
                    "官方链接": str(item.get("url", "") or ""),
                })
        records.append({"股票代码": code, "股票名称": name, "数据状态": "可用", "公告": announcements})

    snapshot = {
        "报告日期": report_date,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "来源": SOURCE_NAME,
        "说明": "仅展示公告标题、日期与官方链接；标题分类不代表利好、利空或投资建议。",
        "stocks": records,
    }
    output_directory.mkdir(exist_ok=True)
    snapshot_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_file

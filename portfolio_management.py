"""本地持仓账本：仅保存用户主动录入的账户、持仓与现金，不参与量化研究。"""

import json
import os
from copy import deepcopy
from pathlib import Path


PORTFOLIO_VERSION = "1.0"
EMPTY_PORTFOLIO = {"version": PORTFOLIO_VERSION, "holdings": [], "cash": []}


def _clean_text(value, field_name, required=False):
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{field_name}不能为空。")
    return text


def _clean_code(value):
    code = _clean_text(value, "股票代码", required=True).zfill(6)
    if not code.isdigit() or len(code) != 6 or code[0] not in {"0", "3", "6"}:
        raise ValueError("股票代码必须是沪深 A 股六位数字代码。")
    return code


def _clean_number(value, field_name, allow_zero=True, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{field_name}必须是数字。")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是数字。") from error
    if number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{field_name}必须{'大于' if not allow_zero else '大于或等于'} 0。")
    if integer and not number.is_integer():
        raise ValueError(f"{field_name}必须是整数。")
    return int(number) if integer else number


def _normalize_holding(item):
    if not isinstance(item, dict):
        raise ValueError("持仓记录格式错误。")
    return {
        "account": _clean_text(item.get("account"), "账户", required=True),
        "code": _clean_code(item.get("code")),
        "name": _clean_text(item.get("name"), "股票名称", required=True),
        "quantity": _clean_number(item.get("quantity"), "持仓数量", allow_zero=False, integer=True),
        "cost_price": _clean_number(item.get("cost_price"), "平均成本"),
        "category": _clean_text(item.get("category"), "类别"),
    }


def _normalize_cash(item):
    if not isinstance(item, dict):
        raise ValueError("现金记录格式错误。")
    return {
        "account": _clean_text(item.get("account"), "账户", required=True),
        "amount": _clean_number(item.get("amount"), "现金余额"),
    }


def load_portfolio(portfolio_file):
    """读取本地账本；文件不存在时返回空账本，不创建文件。"""
    path = Path(portfolio_file)
    if not path.is_file():
        return deepcopy(EMPTY_PORTFOLIO)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取本地持仓账本：{error}。") from error
    if not isinstance(data, dict) or data.get("version") != PORTFOLIO_VERSION:
        raise ValueError("本地持仓账本版本或格式不支持。")
    holdings = data.get("holdings")
    cash = data.get("cash")
    if not isinstance(holdings, list) or not isinstance(cash, list):
        raise ValueError("本地持仓账本缺少 holdings 或 cash 列表。")
    return {
        "version": PORTFOLIO_VERSION,
        "holdings": [_normalize_holding(item) for item in holdings],
        "cash": [_normalize_cash(item) for item in cash],
    }


def save_portfolio(portfolio_file, portfolio):
    """原子保存本地账本，避免半写入；调用方负责确保文件位于本地忽略目录。"""
    path = Path(portfolio_file)
    normalized = {
        "version": PORTFOLIO_VERSION,
        "holdings": [_normalize_holding(item) for item in portfolio.get("holdings", [])],
        "cash": [_normalize_cash(item) for item in portfolio.get("cash", [])],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_file, path)
    return normalized


def upsert_holding(portfolio, holding):
    """按账户与股票代码覆盖保存一条持仓；不把多笔交易误当成已实现收益。"""
    normalized = _normalize_holding(holding)
    updated = deepcopy(portfolio)
    holdings = [_normalize_holding(item) for item in updated.get("holdings", [])]
    replacement_index = next(
        (
            index for index, item in enumerate(holdings)
            if item["account"] == normalized["account"] and item["code"] == normalized["code"]
        ),
        None,
    )
    if replacement_index is None:
        holdings.append(normalized)
    else:
        holdings[replacement_index] = normalized
    updated["version"] = PORTFOLIO_VERSION
    updated["holdings"] = holdings
    updated["cash"] = [_normalize_cash(item) for item in updated.get("cash", [])]
    return updated


def remove_holding(portfolio, account, code):
    """删除用户明确指定的本地持仓记录。"""
    clean_account = _clean_text(account, "账户", required=True)
    clean_code = _clean_code(code)
    updated = deepcopy(portfolio)
    original = [_normalize_holding(item) for item in updated.get("holdings", [])]
    remaining = [item for item in original if not (item["account"] == clean_account and item["code"] == clean_code)]
    if len(remaining) == len(original):
        raise ValueError("未找到要删除的持仓记录。")
    updated["version"] = PORTFOLIO_VERSION
    updated["holdings"] = remaining
    updated["cash"] = [_normalize_cash(item) for item in updated.get("cash", [])]
    return updated


def upsert_cash(portfolio, account, amount):
    """按账户保存现金余额；现金不会参与股票研究或日报。"""
    clean_account = _clean_text(account, "账户", required=True)
    clean_amount = _clean_number(amount, "现金余额")
    updated = deepcopy(portfolio)
    cash = [_normalize_cash(item) for item in updated.get("cash", [])]
    index = next((index for index, item in enumerate(cash) if item["account"] == clean_account), None)
    record = {"account": clean_account, "amount": clean_amount}
    if index is None:
        cash.append(record)
    else:
        cash[index] = record
    updated["version"] = PORTFOLIO_VERSION
    updated["holdings"] = [_normalize_holding(item) for item in updated.get("holdings", [])]
    updated["cash"] = cash
    return updated


def build_portfolio_rows(portfolio, local_quotes, signal_stocks=None):
    """将本地持仓与同日量化快照合并；缺报价时不估算市值或收益。"""
    quote_by_code = local_quotes if isinstance(local_quotes, dict) else {}
    signal_by_code = {
        str(item.get("股票代码", "")).zfill(6): item
        for item in (signal_stocks or [])
        if isinstance(item, dict) and str(item.get("股票代码", "")).strip()
    }
    rows = []
    for holding in portfolio.get("holdings", []):
        item = _normalize_holding(holding)
        quote = quote_by_code.get(item["code"], {})
        close = quote.get("close") if isinstance(quote, dict) else None
        valid_close = isinstance(close, (int, float)) and not isinstance(close, bool) and close >= 0
        market_value = round(item["quantity"] * close, 2) if valid_close else None
        cost_value = round(item["quantity"] * item["cost_price"], 2)
        floating_profit = round(market_value - cost_value, 2) if market_value is not None else None
        floating_return = (
            round(floating_profit / cost_value * 100, 4)
            if floating_profit is not None and cost_value > 0 else None
        )
        signal = signal_by_code.get(item["code"], {})
        rows.append(
            {
                "账户": item["account"],
                "股票代码": item["code"],
                "股票名称": item["name"],
                "类别": item["category"] or "未分类",
                "持仓数量": item["quantity"],
                "平均成本": item["cost_price"],
                "本地最近收盘": close if valid_close else None,
                "行情日期": quote.get("date", "数据不足") if isinstance(quote, dict) else "数据不足",
                "持仓成本": cost_value,
                "当前市值": market_value,
                "浮盈亏": floating_profit,
                "浮盈亏率": floating_return,
                "研究观察": signal.get("信号分类") or quote.get("advice", "数据不足"),
            }
        )
    return sorted(rows, key=lambda row: (row["账户"], row["股票代码"]))


def summarize_portfolio(rows, portfolio):
    """汇总仅由本地账本与本地报价组成的持仓统计。"""
    cost_value = sum(row["持仓成本"] for row in rows)
    market_values = [row["当前市值"] for row in rows if row["当前市值"] is not None]
    floating_profits = [row["浮盈亏"] for row in rows if row["浮盈亏"] is not None]
    cash_total = sum(_normalize_cash(item)["amount"] for item in portfolio.get("cash", []))
    return {
        "持仓数量": len(rows),
        "持仓成本": round(cost_value, 2),
        "已报价持仓市值": round(sum(market_values), 2),
        "已报价浮盈亏": round(sum(floating_profits), 2),
        "现金余额": round(cash_total, 2),
        "缺少本地报价数": len(rows) - len(market_values),
    }


def build_investment_review(rows, signal_stocks=None):
    """生成持仓每日核对清单，只提示事实缺口和已有研究观察，不给出交易指令。"""
    signal_by_code = {
        str(item.get("股票代码", "")).zfill(6): item
        for item in (signal_stocks or [])
        if isinstance(item, dict) and str(item.get("股票代码", "")).strip()
    }
    priced_total = sum(row["当前市值"] for row in rows if row.get("当前市值") is not None)
    review_rows = []
    for row in rows:
        signal = signal_by_code.get(row["股票代码"], {})
        observation_points = signal.get("观察重点", []) if isinstance(signal, dict) else []
        observation = next(
            (str(item) for item in observation_points if isinstance(item, str) and item.strip()),
            "未找到同日研究观察点。",
        )
        signal_label = signal.get("信号分类") or row.get("研究观察") or "数据不足"
        risk_label = signal.get("当前指标", {}).get("风险标签") if isinstance(signal.get("当前指标"), dict) else None
        if row.get("当前市值") is None:
            priority, reason = 1, "未匹配本地量化快照，市值与研究观察不能完整核对。"
        elif signal_label == "偏弱":
            priority, reason = 2, "同日研究信号偏弱；复核原有研究逻辑与已设观察条件。"
        elif risk_label and risk_label != "正常":
            priority, reason = 3, f"同日快照的风险标签为“{risk_label}”。"
        else:
            priority, reason = 4, "暂无额外数据缺口；继续核对下方观察点。"
        market_value = row.get("当前市值")
        weight = round(market_value / priced_total * 100, 2) if market_value is not None and priced_total else None
        review_rows.append(
            {
                "_priority": priority,
                "账户": row["账户"],
                "股票代码": row["股票代码"],
                "股票名称": row["股票名称"],
                "持仓占比": weight,
                "研究观察": signal_label,
                "今日待核对": reason,
                "下一观察点": observation,
            }
        )
    review_rows.sort(key=lambda item: (item["_priority"], -(item["持仓占比"] or 0), item["股票代码"]))
    for item in review_rows:
        item.pop("_priority")
    return review_rows

"""用户确认后按快照收盘价建立本地模拟仓，不连接券商。"""

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path


SIMULATOR_VERSION = "1.0"
COMMISSION_RATE = 0.0003
MINIMUM_COMMISSION = 5.0
LOT_SIZE = 100
EMPTY_SIMULATOR = {"version": SIMULATOR_VERSION, "cash": [], "positions": [], "transactions": []}


def _text(value, field, required=False):
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{field}不能为空。")
    return result


def _number(value, field, positive=False, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{field}必须是数字。")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}必须是数字。") from error
    if result < 0 or (positive and result <= 0):
        raise ValueError(f"{field}必须大于{'0' if positive else '或等于0'}。")
    if integer and not result.is_integer():
        raise ValueError(f"{field}必须是整数。")
    return int(result) if integer else result


def _code(value):
    result = _text(value, "股票代码", required=True).zfill(6)
    if not result.isdigit() or len(result) != 6 or result[0] not in {"0", "3", "6"}:
        raise ValueError("股票代码必须是沪深A股六位数字代码。")
    return result


def _normalize_cash(item):
    if not isinstance(item, dict):
        raise ValueError("模拟现金记录格式错误。")
    return {"account": _text(item.get("account"), "模拟账户", True), "amount": _number(item.get("amount"), "现金余额")}


def _normalize_position(item):
    if not isinstance(item, dict):
        raise ValueError("模拟持仓记录格式错误。")
    return {
        "account": _text(item.get("account"), "模拟账户", True),
        "code": _code(item.get("code")),
        "name": _text(item.get("name"), "股票名称", True),
        "quantity": _number(item.get("quantity"), "模拟持仓数量", True, True),
        "cost_price": _number(item.get("cost_price"), "模拟成本", True),
        "strategy_id": _text(item.get("strategy_id"), "策略标识", True),
        "opened_at": _text(item.get("opened_at"), "建立时间", True),
    }


def load_simulator(simulator_file):
    path = Path(simulator_file)
    if not path.is_file():
        return deepcopy(EMPTY_SIMULATOR)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取本地模拟账本：{error}。") from error
    if not isinstance(data, dict) or data.get("version") != SIMULATOR_VERSION:
        raise ValueError("本地模拟账本版本或格式不支持。")
    cash, positions, transactions = data.get("cash"), data.get("positions"), data.get("transactions")
    if not isinstance(cash, list) or not isinstance(positions, list) or not isinstance(transactions, list):
        raise ValueError("本地模拟账本缺少必要列表。")
    return {"version": SIMULATOR_VERSION, "cash": [_normalize_cash(item) for item in cash], "positions": [_normalize_position(item) for item in positions], "transactions": transactions}


def save_simulator(simulator_file, simulator):
    path = Path(simulator_file)
    normalized = {
        "version": SIMULATOR_VERSION,
        "cash": [_normalize_cash(item) for item in simulator.get("cash", [])],
        "positions": [_normalize_position(item) for item in simulator.get("positions", [])],
        "transactions": list(simulator.get("transactions", [])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return normalized


def upsert_simulator_cash(simulator, account, amount):
    updated = deepcopy(simulator)
    record = {"account": _text(account, "模拟账户", True), "amount": _number(amount, "现金余额")}
    cash = [_normalize_cash(item) for item in updated.get("cash", [])]
    index = next((index for index, item in enumerate(cash) if item["account"] == record["account"]), None)
    if index is None:
        cash.append(record)
    else:
        cash[index] = record
    updated["version"], updated["cash"] = SIMULATOR_VERSION, cash
    updated["positions"] = [_normalize_position(item) for item in updated.get("positions", [])]
    updated["transactions"] = list(updated.get("transactions", []))
    return updated


def create_snapshot_buy(simulator, account, candidate, quantity, strategy_id, snapshot_date):
    """按用户确认的本地快照收盘价建立模拟仓；不可冒充下一日实际成交。"""
    updated = deepcopy(simulator)
    clean_account = _text(account, "模拟账户", True)
    clean_quantity = _number(quantity, "模拟数量", True, True)
    if clean_quantity % LOT_SIZE:
        raise ValueError(f"模拟数量必须为{LOT_SIZE}股的整数倍。")
    if not isinstance(candidate, dict):
        raise ValueError("候选记录格式错误。")
    price = _number(candidate.get("现价"), "快照价格", True)
    code, name = _code(candidate.get("股票代码")), _text(candidate.get("股票名称"), "股票名称", True)
    cash = [_normalize_cash(item) for item in updated.get("cash", [])]
    cash_index = next((index for index, item in enumerate(cash) if item["account"] == clean_account), None)
    if cash_index is None:
        raise ValueError("请先为该模拟账户录入现金余额。")
    gross_value = round(price * clean_quantity, 2)
    commission = round(max(gross_value * COMMISSION_RATE, MINIMUM_COMMISSION), 2)
    total_cost = round(gross_value + commission, 2)
    if cash[cash_index]["amount"] < total_cost:
        raise ValueError("模拟账户现金不足以覆盖成交额和手续费。")
    cash[cash_index]["amount"] = round(cash[cash_index]["amount"] - total_cost, 2)
    positions = [_normalize_position(item) for item in updated.get("positions", [])]
    index = next((i for i, item in enumerate(positions) if item["account"] == clean_account and item["code"] == code), None)
    opened_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if index is None:
        positions.append({"account": clean_account, "code": code, "name": name, "quantity": clean_quantity, "cost_price": round(total_cost / clean_quantity, 4), "strategy_id": _text(strategy_id, "策略标识", True), "opened_at": opened_at})
    else:
        existing = positions[index]
        combined_quantity = existing["quantity"] + clean_quantity
        combined_cost = existing["cost_price"] * existing["quantity"] + total_cost
        existing.update({"quantity": combined_quantity, "cost_price": round(combined_cost / combined_quantity, 4), "strategy_id": _text(strategy_id, "策略标识", True), "opened_at": opened_at})
    transaction = {"时间": opened_at, "类型": "按快照收盘价建立模拟仓", "账户": clean_account, "股票代码": code, "股票名称": name, "策略": _text(strategy_id, "策略标识", True), "快照日期": _text(snapshot_date, "快照日期", True), "价格": price, "数量": clean_quantity, "成交额": gross_value, "手续费": commission, "总成本": total_cost}
    updated.update({"version": SIMULATOR_VERSION, "cash": cash, "positions": positions, "transactions": [transaction] + list(updated.get("transactions", []))})
    return updated, transaction


def summarize_simulator(simulator, quote_map):
    positions = [_normalize_position(item) for item in simulator.get("positions", [])]
    rows, market_values = [], []
    for position in positions:
        quote = quote_map.get(position["code"], {}) if isinstance(quote_map, dict) else {}
        close = quote.get("close") if isinstance(quote, dict) else None
        valid_price = isinstance(close, (int, float)) and close >= 0
        value = round(position["quantity"] * close, 2) if valid_price else None
        cost = round(position["quantity"] * position["cost_price"], 2)
        rows.append({"账户": position["account"], "股票代码": position["code"], "股票名称": position["name"], "策略": position["strategy_id"], "数量": position["quantity"], "模拟成本": position["cost_price"], "本地最近收盘": close if valid_price else None, "行情日期": quote.get("date", "数据不足") if isinstance(quote, dict) else "数据不足", "模拟市值": value, "浮动盈亏": round(value - cost, 2) if value is not None else None})
        if value is not None:
            market_values.append(value)
    cash_total = sum(item["amount"] for item in [_normalize_cash(item) for item in simulator.get("cash", [])])
    return rows, {"模拟持仓数": len(rows), "模拟现金": round(cash_total, 2), "已报价模拟市值": round(sum(market_values), 2), "缺少报价数": len(rows) - len(market_values)}

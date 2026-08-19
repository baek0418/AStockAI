"""AStockAI 面向用户的统一命令行应用入口。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import stock_analysis
from logger import get_logger
from pipeline import run_pipeline
from update_data import run_update_data
from watchlist_manager import (
    append_stock,
    create_stock_config,
    find_watchlist_entry,
    get_watchlist_entries,
    get_watchlist_statistics,
    load_watchlist_data,
    make_editable_stock,
    parse_tags,
    remove_stock,
    replace_stock,
    save_watchlist_data,
    validate_optional_price,
    validate_priority,
    validate_stock_name,
)


def print_banner():
    """输出 AStockAI 命令行应用标题。"""
    print("========================================")
    print("              AStockAI")
    print("        个人 AI 投研助手")
    print("========================================")


def print_main_menu():
    """输出主菜单选项。"""
    print()
    print("1. 一键更新并生成日报")
    print("2. 仅更新行情")
    print("3. 使用已有行情生成研究与日报")
    print("4. 单股票分析")
    print("5. 查看关注股票")
    print("6. 添加关注股票")
    print("7. 修改关注股票")
    print("8. 启用或停用关注股票")
    print("9. 删除关注股票")
    print("10. 查看最近运行状态")
    print()
    print("0. 退出")


def read_menu_choice():
    """读取用户菜单输入并保留为字符串以便校验。"""
    return input("\n请选择：").strip()


def confirm_action(message):
    """请求用户确认操作，仅接受 y、yes、是作为确认。"""
    answer = input(f"{message}（y/n）：").strip().lower()
    return answer in {"y", "yes", "是"}


def get_log_file():
    """返回当天统一日志文件路径。"""
    return Path(__file__).parent / "logs" / f"astockai_{datetime.now().strftime('%Y-%m-%d')}.log"


def print_separator():
    """输出命令行内容分隔线。"""
    print("----------------------------------------")


def print_table(headers, rows):
    """使用标准库字符串格式化输出简单文本表格。"""
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row):
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in string_rows:
        print(format_row(row))


def print_stock_config(stock):
    """展示一只关注股票的完整已知配置字段。"""
    print(f"股票代码：{stock['code']}")
    print(f"股票名称：{stock['name']}")
    print(f"别名：{stock.get('alias', '') or '未设置'}")
    print(f"优先级：{stock.get('priority', 3)}")
    print(f"状态：{'启用' if stock.get('enable', True) else '停用'}")
    print(f"标签：{'、'.join(stock.get('tags', [])) or '未设置'}")
    print(f"持仓成本：{stock.get('cost_price') if stock.get('cost_price') is not None else '未设置'}")
    print(f"目标价：{stock.get('target_price') if stock.get('target_price') is not None else '未设置'}")
    print(f"备注：{stock.get('notes', '') or '未设置'}")


def display_pipeline_result(pipeline_result):
    """从真实 pipeline 返回值展示关键运行结果。"""
    step_lookup = {step["name"]: step for step in pipeline_result["steps"]}
    update_step = step_lookup.get("行情更新", {})
    research_step = step_lookup.get("量化研究", {})
    watchlist_step = step_lookup.get("关注快照", {})
    daily_step = step_lookup.get("每日日报", {})

    print_separator()
    print(f"整体状态：{pipeline_result.get('status', '未知')}")
    print(f"行情更新：{update_step.get('status', '未知')}")
    update_details = update_step.get("details", {})
    print(f"成功股票：{update_details.get('success_count', 0)}")
    print(f"失败股票：{update_details.get('failed_count', 0)}")
    print(f"量化快照状态：{research_step.get('status', '未知')}")
    print(f"量化快照：{research_step.get('output_file') or '无'}")
    print(f"关注快照状态：{watchlist_step.get('status', '未知')}")
    watchlist_details = watchlist_step.get("details", {})
    print(f"匹配：{watchlist_details.get('matched', 0)}")
    print(f"缺失：{watchlist_details.get('missing', 0)}")
    print(f"日报：{daily_step.get('output_file') or '无'}")
    ai_unavailable = daily_step.get("details", {}).get("ai_unavailable")
    print(f"AI 状态：{'不可用，已使用规则分析' if ai_unavailable else '已启用或未使用'}")
    print(f"总耗时：{pipeline_result.get('duration', 0):.1f} 秒")
    print(f"日志：{get_log_file()}")


def run_full_pipeline(logger):
    """确认后调用完整 pipeline，并展示真实结果。"""
    if not confirm_action("即将更新行情并生成最新日报，是否继续"):
        print("已取消操作。")
        logger.info("用户取消完整流水线。")
        return None

    logger.info("CLI 开始执行完整流水线。")
    pipeline_result = run_pipeline()
    display_pipeline_result(pipeline_result)
    logger.info("CLI 完整流水线完成：%s", pipeline_result.get("status"))
    return pipeline_result


def run_update_only(logger):
    """调用既有更新模块，并展示成功、失败和空数据保护情况。"""
    if not confirm_action("即将更新股票行情，是否继续"):
        print("已取消操作。")
        logger.info("用户取消行情更新。")
        return None

    logger.info("CLI 开始执行行情更新。")
    update_result = run_update_data()
    details = update_result.get("details", {}) if isinstance(update_result, dict) else {}
    success_stocks = details.get("success_stocks", [])
    failed_stocks = details.get("failed_stocks", [])
    print_separator()
    print(f"总股票数量：{len(success_stocks) + len(failed_stocks)}")
    print(f"成功数量：{len(success_stocks)}")
    print(f"失败数量：{len(failed_stocks)}")
    print("空数据保护：更新模块不会保存空 CSV。")

    if failed_stocks:
        print("失败股票：")
        for stock in failed_stocks:
            print(f"- {stock.get('name')}（{stock.get('code')}）：{stock.get('error')}")

    print(f"日志：{get_log_file()}")
    logger.info("CLI 行情更新完成：%s", update_result.get("status", "unknown"))
    return update_result


def run_research_report(logger):
    """使用已有 CSV 调用 pipeline 的跳过更新模式生成研究和日报。"""
    print("本操作不会下载最新行情，将使用 data/ 中已有 CSV。")
    if not confirm_action("是否继续生成研究与日报"):
        print("已取消操作。")
        logger.info("用户取消使用已有行情生成研究与日报。")
        return None

    logger.info("CLI 开始使用已有行情生成研究与日报。")
    pipeline_result = run_pipeline(skip_update=True)
    display_pipeline_result(pipeline_result)
    logger.info("CLI 已有行情流程完成：%s", pipeline_result.get("status"))
    return pipeline_result


def load_analysis_record(stock_query):
    """复用单股票分析模块读取已有记录并返回匹配结果与候选项。"""
    output_directory = Path(__file__).parent / "output"
    quant_snapshot = stock_analysis.load_quant_snapshot(output_directory)
    watchlist_snapshot = stock_analysis.load_watchlist_snapshot(output_directory)
    stock_records = stock_analysis.create_stock_records(quant_snapshot, watchlist_snapshot)
    stock_record, error_message = stock_analysis.find_stock(stock_query, stock_records)
    candidates = stock_analysis.find_fuzzy_matches(stock_query, stock_records)
    return stock_record, error_message, candidates


def run_single_stock_analysis(logger, stock_query=None):
    """调用既有单股票分析模块，并展示报告的关键事实。"""
    query = stock_query or input("请输入股票代码或名称：").strip()

    try:
        stock_record, error_message, candidates = load_analysis_record(query)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        logger.warning("单股票分析无法读取快照：%s", error)
        return None

    if error_message:
        if candidates and len(candidates) > 1:
            print("找到多个股票，请输入完整名称或代码：")
            print_table(
                ["股票代码", "股票名称", "别名"],
                [[item["code"], item["name"], item["alias"] or "-"] for item in candidates],
            )
        else:
            print("未在当前量化股票池中找到该股票。")
            print("请先将其加入关注列表并运行行情更新。")
        logger.info("单股票分析未匹配：%s", query)
        return None

    report_file = stock_analysis.run_stock_analysis(query)
    if report_file is None:
        logger.warning("单股票分析报告生成失败：%s", query)
        return None

    fact_snapshot = stock_analysis.build_fact_snapshot(stock_record)
    report_content = Path(report_file).read_text(encoding="utf-8")
    ai_unavailable = "AI增强分析暂不可用" in report_content
    print_separator()
    print(f"股票名称：{fact_snapshot['股票名称']}")
    print(f"股票代码：{fact_snapshot['股票代码']}")
    print(f"Score：{fact_snapshot['综合评分']}")
    print(f"趋势：{fact_snapshot['趋势']}")
    print(f"建议：{fact_snapshot['建议']}")
    print(f"报告文件：{report_file}")
    print(f"AI 状态：{'不可用，已使用规则分析' if ai_unavailable else '已启用'}")
    logger.info("单股票分析完成：%s（%s）", fact_snapshot["股票名称"], fact_snapshot["股票代码"])
    return report_file


def show_watchlist(logger):
    """按配置原始顺序展示启用和停用的全部关注股票。"""
    try:
        watchlist_data = load_watchlist_data()
        entries = get_watchlist_entries(watchlist_data)
        statistics = get_watchlist_statistics(watchlist_data)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        logger.warning("查看关注列表失败：%s", error)
        return None

    rows = [
        [
            entry["index"],
            "启用" if entry["enable"] else "停用",
            entry["priority"],
            entry["code"],
            entry["name"],
            entry["alias"] or "-",
            "、".join(entry["tags"]) or "-",
        ]
        for entry in entries
    ]
    print_table(["序号", "状态", "优先级", "股票代码", "股票名称", "别名", "标签"], rows)
    print(f"总数：{statistics['total']}，启用：{statistics['enabled']}，停用：{statistics['disabled']}")
    logger.info("查看关注列表：总数 %s。", statistics["total"])
    return entries


def select_watchlist_entry(watchlist_data):
    """提示用户定位关注股票，并在匹配多个对象时展示候选项。"""
    query = input("请输入序号、股票代码或名称：").strip()
    entry, error_message, candidates = find_watchlist_entry(watchlist_data, query)

    if error_message:
        print(error_message)
        if candidates:
            print_table(
                ["序号", "股票代码", "股票名称", "别名"],
                [[item["index"], item["code"], item["name"], item["alias"] or "-"] for item in candidates],
            )
        return None

    return entry


def read_stock_inputs(existing_stock=None):
    """读取新增或编辑所需的已知配置字段，回车可保留现有值。"""
    existing_stock = existing_stock or {}
    name = input(f"股票名称 [{existing_stock.get('name', '')}]：").strip() or existing_stock.get("name", "")
    alias = input(f"别名 [{existing_stock.get('alias', '')}]：").strip()
    if not alias and existing_stock:
        alias = existing_stock.get("alias", "")

    priority_text = input(f"优先级 [{existing_stock.get('priority', 3)}]：").strip()
    priority = priority_text or existing_stock.get("priority", 3)
    tags_text = input(f"标签，多个标签用逗号分隔 [{'、'.join(existing_stock.get('tags', []))}]：").strip()
    tags = parse_tags(tags_text) if tags_text else existing_stock.get("tags", [])
    cost_text = input(f"持仓成本 [{existing_stock.get('cost_price') if existing_stock.get('cost_price') is not None else ''}]：").strip()
    cost_price = cost_text if cost_text else existing_stock.get("cost_price")
    target_text = input(f"目标价 [{existing_stock.get('target_price') if existing_stock.get('target_price') is not None else ''}]：").strip()
    target_price = target_text if target_text else existing_stock.get("target_price")
    notes = input(f"备注 [{existing_stock.get('notes', '')}]：").strip()
    if not notes and existing_stock:
        notes = existing_stock.get("notes", "")

    return {
        "name": validate_stock_name(name),
        "alias": alias,
        "priority": validate_priority(priority),
        "tags": parse_tags(tags),
        "cost_price": validate_optional_price(cost_price, "持仓成本"),
        "target_price": validate_optional_price(target_price, "目标价"),
        "notes": notes,
    }


def add_watchlist_stock(logger):
    """交互式添加关注股票，重复代码仅允许重新启用。"""
    try:
        watchlist_data = load_watchlist_data()
        code = input("股票代码：").strip()
        entry, _, _ = find_watchlist_entry(watchlist_data, code)
        if entry:
            if not entry["enable"] and confirm_action("该股票已停用，是否重新启用"):
                stock_config = make_editable_stock(entry)
                stock_config["enable"] = True
                updated_data = replace_stock(watchlist_data, entry["index"], stock_config)
                backup_file = save_watchlist_data(updated_data)
                print(f"已重新启用 {entry['name']}。备份文件：{backup_file}")
                logger.info("重新启用关注股票：%s（%s）。", entry["name"], entry["code"])
                return entry

            print("该股票代码已存在，未重复添加。")
            return None

        input_data = read_stock_inputs()
        stock_config = create_stock_config(code, **input_data)
        print_separator()
        print_stock_config(stock_config)
        if not confirm_action("确认添加该股票"):
            print("已取消操作。")
            logger.info("用户取消添加关注股票：%s。", code)
            return None

        updated_data = append_stock(watchlist_data, stock_config)
        backup_file = save_watchlist_data(updated_data)
    except (FileNotFoundError, ValueError) as error:
        print(f"添加失败：{error}")
        logger.warning("添加关注股票失败：%s", error)
        return None

    print("该股票已加入关注列表。")
    print("需要运行“更新行情”后才会获得量化数据。")
    print(f"备份文件：{backup_file}")
    logger.info("添加关注股票：%s（%s）。", stock_config["name"], stock_config["code"])
    return stock_config


def edit_watchlist_stock(logger):
    """交互式修改已知配置字段，并保留未来未知字段。"""
    try:
        watchlist_data = load_watchlist_data()
        entry = select_watchlist_entry(watchlist_data)
        if entry is None:
            return None

        print("当前配置：")
        print_stock_config(entry)
        input_data = read_stock_inputs(entry)
        updated_stock = make_editable_stock(entry)
        updated_stock.update(input_data)
        updated_stock["enable"] = entry["enable"]
        print("修改后配置：")
        print_stock_config(updated_stock)
        if not confirm_action("确认保存修改"):
            print("已取消操作。")
            logger.info("用户取消修改关注股票：%s。", entry["code"])
            return None

        updated_data = replace_stock(watchlist_data, entry["index"], updated_stock)
        backup_file = save_watchlist_data(updated_data)
    except (FileNotFoundError, ValueError) as error:
        print(f"修改失败：{error}")
        logger.warning("修改关注股票失败：%s", error)
        return None

    print(f"修改成功。备份文件：{backup_file}")
    logger.info("修改关注股票：%s（%s）。", updated_stock["name"], updated_stock["code"])
    return updated_stock


def toggle_watchlist_stock(logger):
    """交互式启用或停用指定关注股票，不删除已有历史文件。"""
    try:
        watchlist_data = load_watchlist_data()
        entry = select_watchlist_entry(watchlist_data)
        if entry is None:
            return None

        target_status = not entry["enable"]
        action_name = "重新启用" if target_status else "停用"
        print(f"当前状态：{'启用' if entry['enable'] else '停用'}")
        if not confirm_action(f"是否将其设置为{action_name}"):
            print("已取消操作。")
            return None

        updated_stock = make_editable_stock(entry)
        updated_stock["enable"] = target_status
        updated_data = replace_stock(watchlist_data, entry["index"], updated_stock)
        backup_file = save_watchlist_data(updated_data)
    except (FileNotFoundError, ValueError) as error:
        print(f"操作失败：{error}")
        logger.warning("启用或停用关注股票失败：%s", error)
        return None

    print(f"已{action_name} {updated_stock['name']}。")
    print("停用后不会进入下一次统一股票池更新；已有历史 CSV 不会自动删除。")
    print("重新启用后需再次运行更新流程。")
    print(f"备份文件：{backup_file}")
    logger.info("%s关注股票：%s（%s）。", action_name, updated_stock["name"], updated_stock["code"])
    return updated_stock


def delete_watchlist_stock(logger):
    """通过两次确认删除关注配置，不删除历史数据和报告。"""
    try:
        watchlist_data = load_watchlist_data()
        entry = select_watchlist_entry(watchlist_data)
        if entry is None:
            return None

        print("即将删除以下配置：")
        print_stock_config(entry)
        print("如仅暂时不关注，建议使用“停用”而不是删除。")
        if not confirm_action("确认从关注列表删除"):
            print("已取消操作。")
            return None

        confirmation_code = input(f"请输入 {entry['code']} 确认删除：").strip()
        if confirmation_code != entry["code"]:
            print("确认代码不一致，已取消删除。")
            return None

        updated_data = remove_stock(watchlist_data, entry["index"])
        backup_file = save_watchlist_data(updated_data)
    except (FileNotFoundError, ValueError) as error:
        print(f"删除失败：{error}")
        logger.warning("删除关注股票失败：%s", error)
        return None

    print(f"已从关注列表删除 {entry['name']}。")
    print("历史 CSV 和历史报告未删除。")
    print(f"备份文件：{backup_file}")
    logger.info("删除关注股票：%s（%s）。", entry["name"], entry["code"])
    return entry


def get_latest_file(output_directory, pattern):
    """按文件名排序查找指定模式的最新输出文件。"""
    files = sorted(output_directory.glob(pattern))
    return files[-1] if files else None


def format_file_time(file_path):
    """格式化文件最后修改时间，文件不存在时返回未找到提示。"""
    if not file_path or not file_path.exists():
        return "未找到"

    return datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def show_latest_status(logger):
    """展示最近流水线状态、快照、日报和日志文件信息。"""
    project_directory = Path(__file__).parent
    output_directory = project_directory / "output"
    status_file = output_directory / "pipeline_status.json"
    quant_file = get_latest_file(output_directory, "quant_snapshot_*.json")
    watchlist_file = get_latest_file(output_directory, "watchlist_snapshot_*.json")
    summary_file = get_latest_file(output_directory, "research_summary_*.json")
    report_file = get_latest_file(output_directory, "每日关注股票日报_*.md")

    print_separator()
    if status_file.exists():
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            print(f"最近 Pipeline 状态：{status_data.get('status', '未知')}")
            print(f"运行编号：{status_data.get('run_id', '未知')}")
            print(f"完成时间：{status_data.get('finished_at', '未知')}")
        except (OSError, json.JSONDecodeError) as error:
            print(f"pipeline_status.json 无法读取：{error}")
    else:
        print("最近 Pipeline 状态：未找到 pipeline_status.json")

    print(f"量化快照：{quant_file or '未找到'}")
    if quant_file:
        try:
            quant_data = json.loads(quant_file.read_text(encoding="utf-8"))
            print(f"  修改时间：{format_file_time(quant_file)}")
            print(f"  快照日期：{quant_data.get('快照日期', '未知')}")
            print(f"  股票总数：{quant_data.get('股票数量', '未知')}")
        except (OSError, json.JSONDecodeError):
            print("  内容无法读取。")

    print(f"关注快照：{watchlist_file or '未找到'}")
    if watchlist_file:
        try:
            watchlist_data = json.loads(watchlist_file.read_text(encoding="utf-8"))
            print(f"  修改时间：{format_file_time(watchlist_file)}")
            print(f"  快照日期：{watchlist_data.get('date', '未知')}")
            print(f"  matched：{watchlist_data.get('matched', '未知')}")
            print(f"  missing：{watchlist_data.get('missing', '未知')}")
        except (OSError, json.JSONDecodeError):
            print("  内容无法读取。")

    print(f"研究摘要：{summary_file or '未找到'}")
    print(f"  修改时间：{format_file_time(summary_file)}")
    print(f"每日日报：{report_file or '未找到'}")
    if report_file:
        try:
            report_content = report_file.read_text(encoding="utf-8")
            ai_status = "AI 不可用，使用规则分析" if "AI增强分析暂不可用" in report_content else "包含 AI 增强分析"
            print(f"  修改时间：{format_file_time(report_file)}")
            print(f"  AI 状态：{ai_status}")
        except OSError:
            print("  内容无法读取。")

    print(f"日志文件：{get_log_file()}")
    logger.info("查看最近运行状态。")


def handle_menu_choice(choice, logger):
    """根据菜单编号执行对应 CLI 操作，并返回是否继续菜单。"""
    actions = {
        "1": lambda: run_full_pipeline(logger),
        "2": lambda: run_update_only(logger),
        "3": lambda: run_research_report(logger),
        "4": lambda: run_single_stock_analysis(logger),
        "5": lambda: show_watchlist(logger),
        "6": lambda: add_watchlist_stock(logger),
        "7": lambda: edit_watchlist_stock(logger),
        "8": lambda: toggle_watchlist_stock(logger),
        "9": lambda: delete_watchlist_stock(logger),
        "10": lambda: show_latest_status(logger),
    }

    if choice == "0":
        print("感谢使用 AStockAI。")
        return False

    action = actions.get(choice)
    if action is None:
        print("无效菜单编号，请输入 0 至 10。")
        return True

    action()
    return True


def run_menu():
    """进入循环菜单，并安全处理用户中断和输入流结束。"""
    logger = get_logger("astock")
    print_banner()

    while True:
        try:
            print_main_menu()
            if not handle_menu_choice(read_menu_choice(), logger):
                return 0
        except KeyboardInterrupt:
            print("\n已安全退出 AStockAI。")
            logger.info("用户通过 Ctrl+C 退出 CLI。")
            return 0
        except EOFError:
            print("\n输入流结束，已安全退出 AStockAI。")
            logger.info("输入流结束，CLI 退出。")
            return 0
        except Exception as error:
            print(f"操作失败：{error}")
            logger.exception("CLI 菜单操作异常")


def run_non_interactive_command(arguments, logger):
    """执行 argparse 子命令，并返回适合 shell 使用的退出码。"""
    if arguments.command == "pipeline":
        result = run_pipeline()
        display_pipeline_result(result)
        return 0 if result.get("success") else 1

    if arguments.command == "update":
        result = run_update_data()
        return 0 if result and result.get("success") else 1

    if arguments.command == "report":
        result = run_pipeline(skip_update=True)
        display_pipeline_result(result)
        return 0 if result.get("success") else 1

    if arguments.command == "analyze":
        result = run_single_stock_analysis(logger, arguments.stock)
        return 0 if result else 1

    if arguments.command == "watchlist":
        show_watchlist(logger)
        return 0

    if arguments.command == "status":
        show_latest_status(logger)
        return 0

    return 2


def main():
    """解析子命令；无子命令时进入交互菜单。"""
    parser = argparse.ArgumentParser(description="AStockAI 统一命令行入口。")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("pipeline", help="更新行情并生成日报。")
    subparsers.add_parser("update", help="仅更新行情。")
    subparsers.add_parser("report", help="使用已有行情生成研究与日报。")
    analyze_parser = subparsers.add_parser("analyze", help="生成单股票分析报告。")
    analyze_parser.add_argument("stock", help="股票代码、全名或唯一模糊名称。")
    subparsers.add_parser("watchlist", help="查看关注股票。")
    subparsers.add_parser("status", help="查看最近运行状态。")
    arguments = parser.parse_args()

    if arguments.command is None:
        return run_menu()

    logger = get_logger("astock")
    try:
        return run_non_interactive_command(arguments, logger)
    except KeyboardInterrupt:
        print("\n已安全退出 AStockAI。")
        logger.info("用户中断非交互 CLI 操作。")
        return 0
    except Exception as error:
        print(f"操作失败：{error}")
        logger.exception("非交互 CLI 操作异常")
        return 1


if __name__ == "__main__":
    sys.exit(main())

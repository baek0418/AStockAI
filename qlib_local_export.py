"""将已核验的前复权本地行情导出为 Qlib 二进制数据。

仅导出复权来源核验为 ``qfq`` 的标的；未复权、未核验或失败标的不会混入。
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from market_data_adapter import STANDARD_COLUMNS, build_standard_market_data


QLIB_FIELDS = ["open", "high", "low", "close", "volume"]


def load_verified_qfq_instruments(audit_file):
    """从来源审计中取得唯一核验为前复权的 instrument 集合。"""
    audit = json.loads(Path(audit_file).read_text(encoding="utf-8"))
    return {
        item["market_code"]
        for item in audit["files"]
        if item.get("status") == "verified" and item.get("adjustment") == "qfq"
    }


def write_qlib_source_files(dataset, instruments, output_directory):
    """按标的拆分 Qlib dump_bin 所需的日线 CSV；只写数值行情字段。"""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = dataset[dataset["instrument"].isin(instruments)].copy()
    if selected.empty:
        raise ValueError("没有已核验的前复权数据可导出到 Qlib。")
    skipped = sorted(set(dataset["instrument"].unique()) - set(selected["instrument"].unique()))
    files = []
    for instrument, group in selected.groupby("instrument", sort=True):
        qlib_frame = group[["date", *QLIB_FIELDS]].sort_values("date").copy()
        file_path = output_directory / f"{instrument}.csv"
        qlib_frame.to_csv(file_path, index=False)
        files.append(file_path)
    return files, skipped


def dump_to_qlib_binary(source_directory, qlib_directory):
    """调用固定的上游 Qlib dump_bin，不复制或改造上游实现。"""
    try:
        from external.qlib.scripts.dump_bin import DumpDataAll
    except ImportError as error:
        raise RuntimeError("未找到 external/qlib；请先下载并在 Qlib 隔离环境运行。") from error
    dumper = DumpDataAll(
        data_path=str(source_directory),
        qlib_dir=str(qlib_directory),
        include_fields=",".join(QLIB_FIELDS),
        max_workers=1,
    )
    dumper.dump()


def validate_qlib_binary(qlib_directory, source_files):
    """读取一只标的的 close，确认二进制内容可被 Qlib 加载且与源 CSV 一致。"""
    try:
        import qlib
        from qlib.data import D
    except ImportError as error:
        raise RuntimeError("Qlib 未安装；请使用 Qlib 隔离环境执行。") from error
    source_file = Path(sorted(source_files)[0])
    source = pd.read_csv(source_file)
    instrument = source_file.stem.upper()
    qlib.init(provider_uri=str(Path(qlib_directory).resolve()), region="cn", redis_port=-1)
    loaded = D.features([instrument], ["$close"], start_time=source["date"].min(), end_time=source["date"].max())
    loaded_close = loaded["$close"].reset_index(drop=True)
    source_close = source["close"].reset_index(drop=True)
    # Qlib bin 使用 float32，不能用 pandas 的严格 float64 相等判断。
    import numpy as np

    if len(loaded_close) != len(source_close) or not np.allclose(
        loaded_close.to_numpy(), source_close.to_numpy(), rtol=1e-6, atol=1e-5
    ):
        raise ValueError(f"Qlib 二进制校验失败：{instrument} 的 close 与源 CSV 不一致。")
    return {"instrument": instrument, "rows": int(len(loaded_close)), "status": "passed"}


def run_qlib_export(project_directory=None):
    """生成 Qlib 专用输入与二进制数据，绝不写入 data/、models/ 或现有预测产物。"""
    project = Path(project_directory or Path(__file__).parent)
    report_date = datetime.now().strftime("%Y-%m-%d")
    audit_file = project / "output" / "research" / f"market_data_adjustment_audit_{report_date}.json"
    if not audit_file.exists():
        raise ValueError("缺少当日复权来源审计；拒绝导出 Qlib 数据。")
    dataset, _ = build_standard_market_data(project / "data", project)
    if list(dataset.columns) != STANDARD_COLUMNS:
        raise ValueError("标准行情字段不完整，拒绝导出 Qlib 数据。")
    qfq_instruments = load_verified_qfq_instruments(audit_file)
    research_directory = project / "output" / "research"
    source_directory = research_directory / f"qlib_source_{report_date}"
    qlib_directory = research_directory / f"qlib_data_{report_date}"
    source_files, skipped = write_qlib_source_files(dataset, qfq_instruments, source_directory)
    dump_to_qlib_binary(source_directory, qlib_directory)
    validation = validate_qlib_binary(qlib_directory, source_files)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_adjustment_audit": str(audit_file),
        "selection_rule": "仅导出 status=verified 且 adjustment=qfq 的标的。",
        "selected_instruments": sorted(qfq_instruments),
        "excluded_instruments": skipped,
        "fields": QLIB_FIELDS,
        "source_directory": str(source_directory),
        "qlib_directory": str(qlib_directory),
        "binary_validation": validation,
    }
    report_file = research_directory / f"qlib_export_{report_date}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_file


if __name__ == "__main__":
    result, output_file = run_qlib_export()
    print(output_file)
    print({"selected": len(result["selected_instruments"]), "excluded": len(result["excluded_instruments"]), **result["binary_validation"]})

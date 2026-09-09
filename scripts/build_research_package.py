"""审计探针或从已核验证据发布严格历史研究输入包。"""
import argparse
import json
from pathlib import Path

from astock_core.research.historical_package_builder import audit_probe_directory, build_strict_package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-probe', type=Path)
    parser.add_argument('--candidate-directory', type=Path)
    parser.add_argument('--project-directory', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if bool(args.audit_probe) == bool(args.candidate_directory):
        parser.error('必须且只能指定 --audit-probe 或 --candidate-directory。')
    if args.audit_probe:
        report = audit_probe_directory(args.audit_probe)
        target = args.audit_probe / 'readiness.json'
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(target)
        return 2 if not report['可发布正式包'] else 0
    result = build_strict_package(args.project_directory, args.candidate_directory)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

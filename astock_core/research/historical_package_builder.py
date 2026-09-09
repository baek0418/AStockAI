"""从经过人工/供应方核验的候选文件发布严格历史研究输入包。"""
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from astock_core.research.historical_inputs import INPUT_NAMES, load_historical_inputs, sha256


REQUIRED_DECLARATIONS = (
    'membership_availability_verified',
    'execution_states_complete',
    'corporate_actions_complete',
)


def _read_candidate(source, name):
    path = source / name
    if not path.is_file():
        raise ValueError(f'候选输入缺少 {name}。')
    return pd.read_csv(path, dtype={'股票代码': str})


def build_strict_package(project_directory, candidate_directory):
    """验证独立证据后原子发布；任何缺口都不会产生正式 manifest。"""
    project = Path(project_directory).resolve()
    source = Path(candidate_directory).resolve()
    evidence_file = source / 'evidence.json'
    if not evidence_file.is_file():
        raise ValueError('缺少 evidence.json，不能证明候选数据的时点和完整性。')
    evidence = json.loads(evidence_file.read_text(encoding='utf-8'))
    missing = [key for key in REQUIRED_DECLARATIONS if evidence.get(key) is not True]
    if missing:
        raise ValueError(f'证据声明未通过：{missing}。')
    sources = evidence.get('sources', {})
    if any(not str(sources.get(name, '')).strip() for name in INPUT_NAMES):
        raise ValueError('四类输入都必须声明可复核来源。')

    calendar = _read_candidate(source, 'calendar.csv')
    membership = _read_candidate(source, 'membership.csv')
    raw = _read_candidate(source, 'prices_raw.csv')
    states = _read_candidate(source, 'execution_states.csv')
    actions = _read_candidate(source, 'corporate_actions.csv')
    if 'adjustflag' not in raw or not raw['adjustflag'].astype(str).eq('3').all():
        raise ValueError('候选成交价格必须逐行证明 adjustflag=3（不复权）。')
    state_columns = {'日期','股票代码','可买入','可卖出','可得时间','状态来源'}
    if not state_columns.issubset(states):
        raise ValueError(f'交易状态证据缺少字段：{sorted(state_columns-set(states))}。')
    if states.duplicated(['日期','股票代码']).any():
        raise ValueError('交易状态证据存在重复日期/股票。')
    raw_required = {'日期','股票代码','收盘','成交量'}
    if not raw_required.issubset(raw):
        raise ValueError(f'原始行情缺少字段：{sorted(raw_required-set(raw))}。')
    prices = raw[['日期','股票代码','收盘','成交量']].merge(
        states[['日期','股票代码','可买入','可卖出','可得时间','状态来源']],
        on=['日期','股票代码'], how='left', validate='one_to_one'
    )
    if prices[['可买入','可卖出','可得时间','状态来源']].isna().any().any():
        raise ValueError('并非每条原始行情都有独立、明确的交易状态证据。')
    prices['复权方式'] = 'raw'

    target = project / 'data/research_execution'
    if target.exists():
        raise FileExistsError('正式研究输入目录已存在；请先审计并显式归档，构建器不会覆盖。')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='research_execution_', dir=target.parent) as temporary:
        stage = Path(temporary)
        frames = {'calendar': calendar, 'membership': membership, 'prices': prices,
                  'corporate_actions': actions}
        manifest = {
            'version': 1,
            'universe_size': evidence.get('universe_size'),
            'corporate_actions_complete': True,
            'built_at': datetime.now().astimezone().isoformat(),
            'evidence_sha256': sha256(evidence_file),
            'files': {},
        }
        for name, frame in frames.items():
            path = stage / f'{name}.csv'
            frame.to_csv(path, index=False)
            manifest['files'][name] = {
                'path': path.name, 'source': sources[name], 'sha256': sha256(path),
            }
        (stage / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        # 在临时项目结构中调用正式加载器，确保发布物与回测入口使用同一套校验。
        validation_root = stage / '_validation_root'
        validation_target = validation_root / 'data/research_execution'
        validation_target.mkdir(parents=True)
        for item in stage.glob('*.csv'):
            shutil.copy2(item, validation_target / item.name)
        shutil.copy2(stage / 'manifest.json', validation_target / 'manifest.json')
        _, audit = load_historical_inputs(validation_root)
        shutil.rmtree(validation_root)
        shutil.move(str(stage), str(target))
    return {'status': 'published', 'directory': str(target), 'audit': audit}


def audit_probe_directory(probe_directory):
    """把 BaoStock 探针转换为机器可读的准备度报告，不把探针伪装成正式输入。"""
    root = Path(probe_directory)
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    received = {item['query']: item for item in summary if item.get('status') == 'received'}
    membership_files = sorted(root.glob('membership_*.json'))
    membership_sets = []
    for path in membership_files:
        payload = json.loads(path.read_text(encoding='utf-8'))
        membership_sets.append({row['code'] for row in payload.get('rows', [])})
    changed = None
    if len(membership_sets) >= 2:
        changed = len(membership_sets[0] ^ membership_sets[1]) // 2
    report = {
        '状态': 'staged_incomplete',
        '来源': 'BaoStock 0.9.3 probe',
        '探针目录': str(root.resolve()),
        '已验证能力': {
            '历史成分查询': 'membership' in received,
            '原始行情及停牌标记': 'status' in received,
            '分红事件字段': 'dividend' in received,
            '两期样本各侧成分变化数': changed,
        },
        '阻塞项': [
            '历史成分 updateDate 尚不能证明精确的盘前/盘后可得时间',
            'tradestatus 不提供涨跌停方向的显式可买入/可卖出证据',
            '税后红利可能含多个持有期税率情景，不能自动选择',
            '尚未抓取连续交易日、全体历史成分和全区间公司行为',
        ],
        '可发布正式包': False,
    }
    return report

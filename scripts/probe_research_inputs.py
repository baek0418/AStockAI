"""小样本检查 BaoStock 历史接口，保存原始响应；不自动宣称满足严格回测。"""
import contextlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

QUERIES = [('membership','2024-05-14'),('membership','2026-08-20'),('dividend','2025'),('status','2026-09-08')]


def worker(kind, parameter):
    import baostock as bs
    with contextlib.redirect_stdout(io.StringIO()):
        login=bs.login()
        if login.error_code != '0':
            raise ValueError(login.error_msg)
        try:
            if kind == 'membership':
                result=bs.query_hs300_stocks(parameter)
            elif kind == 'dividend':
                result=bs.query_dividend_data('sh.600519',year=parameter,yearType='operate')
            elif kind == 'status':
                result=bs.query_history_k_data_plus('sh.600519','date,code,close,volume,tradestatus,isST,adjustflag',start_date='2026-09-01',end_date=parameter,adjustflag='3')
            else:
                raise ValueError('未知查询')
            if result.error_code != '0':
                raise ValueError(result.error_msg)
            frame=result.get_data()
            if result.error_code != '0':
                raise ValueError(result.error_msg)
        finally:
            bs.logout()
    print(json.dumps({'fields':list(frame.columns),'rows':frame.to_dict('records')},ensure_ascii=False))


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'output/research_input_probe'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    summary=[]
    for kind,parameter in QUERIES:
        record={'source':'BaoStock 0.9.3','query':kind,'parameter':parameter,'retrieved_at':datetime.now().isoformat()}
        try:
            result=subprocess.run([sys.executable,'-m','scripts.probe_research_inputs','--worker',kind,parameter],cwd=root,capture_output=True,text=True,timeout=25)
            if result.returncode:
                raise ValueError(result.stderr[-1000:])
            payload=json.loads(result.stdout)
            record.update(status='received',**payload)
        except (ValueError,subprocess.TimeoutExpired) as error:
            record.update(status='failed',error=str(error))
        (output/f'{kind}_{parameter}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2), encoding="utf-8")
        summary.append({k:v for k,v in record.items() if k != 'rows'}|{'row_count':len(record.get('rows',[]))})
        print(json.dumps(summary[-1],ensure_ascii=False),flush=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2), encoding="utf-8")
    print(output)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(sys.argv[2],sys.argv[3])
    else:
        main()

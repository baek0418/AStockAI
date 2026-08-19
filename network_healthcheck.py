"""以项目真实 HTTP 客户端检测行情源与 AI 服务的可达性，不输出任何密钥。"""

import json
from datetime import datetime

import requests

from research_universe import CSI300_BOARD_CODE, EASTMONEY_CONSTITUENTS_URL


def probe(name, url, expected_statuses, *, params=None, request_get=requests.get):
    """返回安全、可读的服务连通性结果。"""
    try:
        response = request_get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
    except requests.RequestException as error:
        return {"服务": name, "状态": "不可达", "说明": f"{type(error).__name__}：网络、代理或上游连接未完成。"}
    if response.status_code not in expected_statuses:
        return {"服务": name, "状态": "异常", "HTTP状态": response.status_code, "说明": "服务返回了非预期状态。"}
    return {"服务": name, "状态": "可达", "HTTP状态": response.status_code}


def run_healthcheck():
    """检测股票池下载与 AI 请求使用的两个 HTTPS 目标。"""
    results = [
        probe(
            "沪深300成分股源",
            EASTMONEY_CONSTITUENTS_URL,
            {200},
            params={"pn": 1, "pz": 1, "fid": "f3", "fs": f"b:{CSI300_BOARD_CODE}", "fields": "f12,f14"},
        ),
        # 未携带 Authorization 时 401 代表网络已到达 AI 服务，不验证账户或额度。
        probe("DeepSeek服务", "https://api.deepseek.com", {401, 404}),
    ]
    return {"检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "服务": results, "全部可达": all(item["状态"] == "可达" for item in results)}


def main():
    result = run_healthcheck()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["全部可达"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

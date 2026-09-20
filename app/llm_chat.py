"""页面内多模型直调：豆包 / DeepSeek（均走 OpenAI 兼容协议，火山方舟 / DeepSeek 开放平台）。

学生 AI 助手、教师 AI 工具都在系统网页内直接向所选品牌模型问答，无需跳转到外部网站。
选中的品牌未配 API Key 或调用失败时，自动按回退顺序尝试其它品牌，全部不可用才返回 None（由调用方走演示模板）。
"""
from . import config

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

_ALL = ("doubao", "deepseek")


def supported() -> dict:
    """返回各品牌是否已配置 API Key。"""
    return {
        "doubao": bool(config.DOUBAO_API_KEY),
        "deepseek": bool(config.LLM_API_KEY),
    }


def label(provider: str) -> str:
    return "豆包" if provider == "doubao" else "DeepSeek"


def _endpoint(provider: str):
    if provider == "doubao":
        return config.DOUBAO_BASE_URL, config.DOUBAO_API_KEY, config.DOUBAO_MODEL
    return config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL


def _order(provider) -> list:
    """将用户的选择（字符串或优先级列表）转成尝试顺序，默认 豆包→DeepSeek。"""
    if isinstance(provider, list):
        order = [p for p in provider if p in _ALL]
    elif isinstance(provider, str) and provider in _ALL:
        order = [provider]
    else:
        order = list(_ALL)
    return order or ["deepseek"]


def chat(messages: list, provider="deepseek", temperature: float = 0.6):
    """按顺序尝试各品牌，返回 (text|None, used_provider|None)。"""
    if not _HAS_OPENAI:
        return None, None
    for p in _order(provider):
        url, key, model = _endpoint(p)
        if not key:
            continue
        try:
            client = OpenAI(api_key=key, base_url=url)
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            return resp.choices[0].message.content.strip(), p
        except Exception:
            continue
    return None, None
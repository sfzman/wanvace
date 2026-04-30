"""Prompt translation helper backed by an OpenAI-compatible Chat Completions API."""
from __future__ import annotations

import json
from urllib import error, request

from utils.app_config import get_prompt_translation_config

DEFAULT_TARGET_LANGUAGE = "English"

SYSTEM_PROMPT = """You translate text-to-video prompts into concise, faithful English.

Rules:
1. Output JSON only with keys "prompt" and "negative_prompt".
2. Translate faithfully without adding or removing visual details.
3. Preserve prompt syntax such as commas, parentheses, brackets, weights, and style tokens.
4. Keep empty inputs as empty strings.
5. If the source text is already in the target language, keep it natural and concise.
"""


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_content(response_data: dict) -> str:
    choices = response_data.get("choices") or []
    if not choices:
        raise ValueError("响应中缺少 choices")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "".join(text_parts)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("响应中缺少可解析的文本内容")
    return content


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


def translate_prompts(prompt: str, negative_prompt: str, target_language: str = DEFAULT_TARGET_LANGUAGE):
    """Translate prompt fields with a Chat Completions compatible endpoint."""
    source_prompt = _normalize_text(prompt)
    source_negative_prompt = _normalize_text(negative_prompt)

    if not source_prompt and not source_negative_prompt:
        return source_prompt, source_negative_prompt, "请先输入需要翻译的提示词"

    config = get_prompt_translation_config()
    if not config["api_key"]:
        return source_prompt, source_negative_prompt, "未配置 ARK_CODING_API_KEY，无法调用翻译接口"
    if not config["model"]:
        return source_prompt, source_negative_prompt, "未配置 ARK_CODING_MODEL，无法调用翻译接口"

    payload = {
        "model": config["model"],
        "temperature": 0.2,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": target_language,
                        "prompt": source_prompt,
                        "negative_prompt": source_negative_prompt,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    req = request.Request(
        url=f"{config['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config["timeout"]) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        detail = detail[:240] if detail else exc.reason
        return source_prompt, source_negative_prompt, f"翻译失败：HTTP {exc.code} - {detail}"
    except Exception as exc:  # noqa: BLE001
        return source_prompt, source_negative_prompt, f"翻译失败：{exc}"

    try:
        response_data = json.loads(body)
        content = _strip_code_fences(_extract_content(response_data))
        translated_data = _parse_json_object(content)
    except Exception as exc:  # noqa: BLE001
        return source_prompt, source_negative_prompt, f"翻译失败：响应解析异常 - {exc}"

    translated_prompt = _normalize_text(translated_data.get("prompt"))
    translated_negative_prompt = _normalize_text(translated_data.get("negative_prompt"))

    if source_prompt and not translated_prompt:
        translated_prompt = source_prompt
    if source_negative_prompt and not translated_negative_prompt:
        translated_negative_prompt = source_negative_prompt

    if translated_prompt == source_prompt and translated_negative_prompt == source_negative_prompt:
        status = f"翻译完成：内容未变化（目标语言：{target_language}）"
    else:
        status = f"翻译完成：已转换为{target_language}"

    return translated_prompt, translated_negative_prompt, status

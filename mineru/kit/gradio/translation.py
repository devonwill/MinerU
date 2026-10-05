"""全文翻译：按 Markdown 页调用 OpenAI 兼容大模型，供 Gradio 层后处理使用。

解析流程、api-server 与协议均不感知本模块；翻译失败时给出可定位页码的明确错误。
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# docvortex FULL Markdown 以该分隔符连接各页，按其切回逐页后再逐页翻译。
PAGE_SEPARATOR = "\n\n---\n\n"

# 下拉值同时是传给模型的语言名，key 为界面语言，value 为模型侧规范名称。
LANGUAGES: dict[str, str] = {
    "英语": "English",
    "中文（简体）": "Simplified Chinese",
    "中文（繁体）": "Traditional Chinese",
    "日语": "Japanese",
    "韩语": "Korean",
    "法语": "French",
    "德语": "German",
    "西班牙语": "Spanish",
    "葡萄牙语": "Portuguese",
    "俄语": "Russian",
    "意大利语": "Italian",
    "阿拉伯语": "Arabic",
    "印地语": "Hindi",
    "泰语": "Thai",
    "越南语": "Vietnamese",
    "自动检测": "Auto-detect",
}
DEFAULT_SOURCE_LANGUAGE = "英语"
DEFAULT_TARGET_LANGUAGE = "中文（简体）"
AUTO_DETECT_LANGUAGE = "自动检测"


@dataclass(frozen=True)
class ProviderPreset:
    """一个服务商预设；model 为空时由用户自行填写。"""

    base_url: str
    model: str = ""
    api_key: str = ""


# 预设顺序即界面下拉顺序；选择预设后把 base_url/model/api_key 回填到可编辑控件。
PROVIDERS: dict[str, ProviderPreset] = {
    "本地 Ollama": ProviderPreset(base_url="http://127.0.0.1:11434/v1", model="qwen3.8:27b", api_key="ollama"),
    "小米 MiMo Token Plan": ProviderPreset(
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-flash"
    ),
    "MiniMax M Plan": ProviderPreset(base_url="https://api.minimax.cn/v1", model="MiniMax-M3.1-Flash-Preview"),
    "通义千问": ProviderPreset(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus"),
    "DeepSeek": ProviderPreset(base_url="https://api.deepseek.com/v1", model="deepseek-chat"),
    "OpenAI": ProviderPreset(base_url="https://api.openai.com/v1", model="gpt-4o"),
    "自定义": ProviderPreset(base_url=""),
}
DEFAULT_PROVIDER = "本地 Ollama"

_MAX_RETRIES = 3
# 代码块、行内代码、图片、链接、公式、HTML 标签之外的可读文字少于该长度则跳过该页。
_MIN_NATURAL_TEXT_LEN = 2

_SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.
Translate the user's Markdown text from {source} into {target}.

Rules:
1. Translate ONLY the natural-language content. Output the translation itself,
with no explanations, notes, or preface.
2. Keep all Markdown syntax unchanged: headings (#), bold/italic markers, list
markers, blockquotes (>), tables (|), and link/image syntax [](), ![]().
3. Keep code blocks (```), inline code (`), LaTeX math ($...$, $$...$$), HTML
tags, and URLs exactly as in the source; do not translate or reformat them.
4. Do not add, remove, merge, or reorder paragraphs. Preserve all line breaks
and blank lines.
5. Proper nouns, product names, and formulas may be kept in the original
language when appropriate."""

# 匹配需要原样保留的片段：围栏代码块、行内代码、公式、图片/链接 URL、HTML 标签。
_PRESERVE_PATTERN = re.compile(
    r"```.*?```"
    r"|`[^`\n]*`"
    r"|\$\$.*?\$\$"
    r"|\$[^$\n]+\$"
    r"|!\[[^\]]*\]\([^)]*\)"
    r"|\[[^\]]*\]\([^)]*\)"
    r"|</?[a-zA-Z][^>]*>",
    re.DOTALL,
)
# 自然语言字符：中日韩文字、拉丁字母（含变音符）、西里尔字母等。
_NATURAL_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7afA-Za-zÀ-ÿЀ-ӿ]")


class TranslationError(RuntimeError):
    """翻译配置或单页翻译失败，消息可直接展示给用户。"""


@dataclass(frozen=True)
class TranslationConfig:
    """一次全文翻译所需的完整参数，均来自界面控件。"""

    source_language: str
    target_language: str
    base_url: str
    model: str
    api_key: str
    concurrency: int = 4
    # 单次请求总超时（秒）：卡住时快速失败并重试，避免请求长时间占用模型槽位。
    request_timeout: float = 120.0


def _has_natural_text(page: str) -> bool:
    """剔除代码、公式、图片、标签后判断该页是否含有值得翻译的自然语言。"""
    stripped = _PRESERVE_PATTERN.sub("", page)
    return len(_NATURAL_CHAR_PATTERN.findall(stripped)) >= _MIN_NATURAL_TEXT_LEN


def _build_system_prompt(source: str, target: str) -> str:
    """源语言为自动检测时不向模型指定源语言，其余情况显式指定。"""
    source_text = "the source language" if source == LANGUAGES[AUTO_DETECT_LANGUAGE] else source
    return _SYSTEM_PROMPT_TEMPLATE.format(source=source_text, target=target)


def _translate_page(page: str, config: TranslationConfig, page_number: int) -> str:
    """翻译单页并重试；空页与纯公式/图片页原样返回。

    openai 为第三方库，在工作线程内按需导入，避免模块顶层持有重依赖。
    """
    if not page.strip() or not _has_natural_text(page):
        return page
    from openai import OpenAI

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        # 每次重试都新建客户端，保证连接干净，不继承上次卡住的请求。
        client = OpenAI(base_url=config.base_url, api_key=config.api_key or "not-needed", timeout=config.request_timeout)
        try:
            completion = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": _build_system_prompt(config.source_language, config.target_language)},
                    {"role": "user", "content": page},
                ],
                temperature=0.3,
                stream=False,
            )
            translated = completion.choices[0].message.content
            if not translated or not translated.strip():
                raise TranslationError("模型返回了空译文")
            return translated.strip("\n")
        except Exception as exc:  # noqa: PERF203 - 重试需保留最后一次错误
            last_error = exc
            if attempt < _MAX_RETRIES - 1:
                # 简单指数退避，给被拖慢或正在重载的模型一点恢复时间。
                time.sleep(0.5 * (attempt + 1))
    raise TranslationError(f"第 {page_number} 页翻译失败（已重试 {_MAX_RETRIES} 次）：{last_error}")


def translate_markdown(
    markdown: str,
    config: TranslationConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    show_progress: bool = True,
) -> str:
    """按 Markdown 页切分并翻译，保持页分隔符与页顺序；失败时抛出 TranslationError。

    Args:
        markdown: 原始 FULL 模式 Markdown。
        config: 翻译参数（语言、服务商、模型、并发）。
        progress_callback: 每完成一页回调 (已完成页数, 总页数)，用于界面进度展示。
        show_progress: 是否在命令行输出 tqdm 进度条。
    """
    if not config.base_url.strip():
        raise TranslationError("请填写服务商接口地址（Base URL）")
    if not config.model.strip():
        raise TranslationError("请填写翻译模型名称")
    if config.source_language == config.target_language:
        raise TranslationError("源语言与目标语言相同，无需翻译")
    if config.concurrency < 1:
        raise TranslationError("并发数必须大于 0")
    if config.request_timeout <= 0:
        raise TranslationError("请求超时必须大于 0")

    pages = markdown.split(PAGE_SEPARATOR)
    total = len(pages)
    translated_pages: list[str] = [""] * total
    done = 0

    def worker(index: int) -> tuple[int, str]:
        return index, _translate_page(pages[index], config, index + 1)

    # tqdm 为轻量第三方库，按需导入以便调用方关闭命令行进度展示。
    from tqdm import tqdm

    pbar = tqdm(total=total, desc="Translate", disable=not show_progress) if show_progress else None
    try:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            for index, translated in executor.map(worker, range(total)):
                translated_pages[index] = translated
                done += 1
                if pbar is not None:
                    pbar.update(1)
                if progress_callback is not None:
                    progress_callback(done, total)
    finally:
        if pbar is not None:
            pbar.close()

    return PAGE_SEPARATOR.join(translated_pages)


def test_translation_connection(config: TranslationConfig) -> None:
    """发起一次极简对话请求验证连接与模型可用性；失败时抛出 TranslationError。

    先做与正式翻译一致的配置校验，再用 max_tokens=1 的最小请求探测端点，
    避免连通性测试消耗过多 token；openai 在函数体内惰性导入。
    """
    if not config.base_url.strip():
        raise TranslationError("请填写服务商接口地址（Base URL）")
    if not config.model.strip():
        raise TranslationError("请填写翻译模型名称")
    if config.source_language == config.target_language:
        raise TranslationError("源语言与目标语言相同，无需翻译")

    from openai import OpenAI

    # 连通性探测使用较短超时，避免界面长时间停在"测试中"。
    client = OpenAI(base_url=config.base_url, api_key=config.api_key or "not-needed", timeout=min(30.0, config.request_timeout))
    try:
        completion = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
            stream=False,
        )
    except Exception as exc:
        # 前缀文案由 UI 层用 i18n 统一包装，这里只保留错误原因。
        raise TranslationError(str(exc)) from exc
    if not completion.choices:
        raise TranslationError("模型未返回任何结果")


def render_translation_html(markdown_text: str, *, asset_base_url: str) -> str:
    """把译文 Markdown 渲染为可嵌入 iframe 的独立 HTML。

    译文图片沿用原文图片目录，故以 artifact 目录作为相对链接基准。
    """
    from markdown_it import MarkdownIt

    body = MarkdownIt("commonmark", {"html": True, "linkify": True, "typographer": True}).enable("table").render(markdown_text)
    return f"""<!DOCTYPE html>
<html lang="">
<head>
<meta charset="utf-8">
<base href="{asset_base_url}/">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       line-height: 1.7; padding: 4px 20px; color: #1f2329; }}
img {{ max-width: 100%; height: auto; }}
pre, code {{ font-family: 'SFMono-Regular', Consolas, monospace; }}
pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; }}
blockquote {{ margin: 0; padding-left: 12px; border-left: 3px solid #d0d7de; color: #57606a; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# 译文多格式导出：解析译文 Markdown 为统一节点树，再分别渲染为 JSON/DOCX/LaTeX/PDF。
# MinerU/DocVortex 的渲染器只接受 MiddleJson，无法消费译文 Markdown，故在此独立实现，
# 不回填协议、不引入新依赖（markdown_it / python-docx / reportlab 均为环境已有库）。
# ---------------------------------------------------------------------------


@dataclass
class InlineNode:
    """内联节点：最小标记集合，嵌套 children 用于 strong/em/link。"""

    type: str  # text | strong | em | code | link | image | math | linebreak
    text: str = ""
    href: str = ""  # link 的 URL、image 的路径
    children: list["InlineNode"] = field(default_factory=list)


@dataclass
class BlockNode:
    """块级节点；type 决定内容字段的含义。"""

    type: str  # heading | paragraph | list_item | code | quote | hr | table
    level: int = 0  # heading 层级
    ordered: bool = False  # list_item 是否为有序列表
    children: list["BlockNode"] = field(default_factory=list)  # list_item/quote 内嵌块
    inline: list[InlineNode] = field(default_factory=list)  # heading/paragraph/list_item 的内联内容
    code: str = ""  # code 文本
    language: str = ""  # code 语言
    rows: list[list[list[InlineNode]]] = field(default_factory=list)  # table 各单元格（首行为表头）


def _find_matching_close(tokens, open_index: int, end: int) -> int:  # noqa: ANN001
    """给定 *_open 位置，按嵌套深度找到对应 *_close，返回其索引（找不到则 end-1）。

    同时用于 block 级（heading/list/quote）和 inline 级（strong/em/link）token 流，
    两者都遵循 open/close 配对约定，逻辑完全相同。
    """
    open_type = tokens[open_index].type
    close_type = open_type.replace("_open", "_close")
    depth = 0
    for index in range(open_index, end):
        token_type = tokens[index].type
        if token_type == open_type:
            depth += 1
        elif token_type == close_type:
            depth -= 1
            if depth == 0:
                return index
    return end - 1


def _parse_inline_tokens(tokens, start: int, end: int) -> list[InlineNode]:  # noqa: ANN001
    """把半开区间 [start, end) 内的扁平内联 token 转换为 InlineNode 列表。"""
    nodes: list[InlineNode] = []
    index = start
    while index < end:
        token = tokens[index]
        kind = token.type
        if kind in {"text", "code_inline", "math_inline"}:
            node_type = {"text": "text", "code_inline": "code", "math_inline": "math"}[kind]
            nodes.append(InlineNode(node_type, text=token.content))
            index += 1
        elif kind in {"softbreak", "hardbreak"}:
            nodes.append(InlineNode("linebreak", text="\n"))
            index += 1
        elif kind in {"strong_open", "em_open", "link_open"}:
            close = _find_matching_close(tokens, index, end)
            children = _parse_inline_tokens(tokens, index + 1, close)
            node_type = {"strong_open": "strong", "em_open": "em", "link_open": "link"}[kind]
            href = token.attrs.get("href", "") if kind == "link_open" else ""
            nodes.append(InlineNode(node_type, href=href, children=children))
            index = close + 1
        elif kind == "image":
            nodes.append(InlineNode("image", text=token.content, href=token.attrs.get("src", "")))
            index += 1
        elif kind == "html_inline":
            # 内联 HTML 标签作为纯文本保留，避免裸标签污染导出。
            nodes.append(InlineNode("text", text=token.content))
            index += 1
        else:
            index += 1
    return nodes


def _parse_inline(inline_token) -> list[InlineNode]:  # noqa: ANN001
    """把一个 inline token 的扁平 children 转换为 InlineNode 列表。"""
    children = inline_token.children or []
    return _parse_inline_tokens(children, 0, len(children))


def _parse_list(tokens, start: int, end: int) -> tuple[list[BlockNode], int]:  # noqa: ANN001
    """解析一个 bullet/ordered list 容器，返回 (顶层 list_item 列表, 容器后一索引)。"""
    ordered = tokens[start].type == "ordered_list_open"
    list_close = _find_matching_close(tokens, start, end)
    items: list[BlockNode] = []
    index = start + 1
    while index < list_close:
        if tokens[index].type != "list_item_open":
            index += 1
            continue
        item_close = _find_matching_close(tokens, index, list_close + 1)
        # 列表项 = 首个段落内联内容 + 可能的嵌套列表。
        inline: list[InlineNode] = []
        nested: list[BlockNode] = []
        cursor = index + 1
        while cursor < item_close:
            current = tokens[cursor]
            if current.type == "inline":
                if not inline:
                    inline = _parse_inline(current)
                cursor += 1
            elif current.type in {"bullet_list_open", "ordered_list_open"}:
                sub_items, cursor = _parse_list(tokens, cursor, item_close + 1)
                nested.extend(sub_items)
            else:
                cursor += 1
        items.append(BlockNode("list_item", ordered=ordered, inline=inline, children=nested))
        index = item_close + 1
    return items, list_close + 1


def _parse_blocks(tokens, start: int = 0, end: int | None = None) -> tuple[list[BlockNode], int]:  # noqa: ANN001
    """解析半开区间 [start, end) 内的块级 token，返回 (块节点列表, end)。"""
    if end is None:
        end = len(tokens)
    blocks: list[BlockNode] = []
    index = start
    while index < end:
        token = tokens[index]
        kind = token.type
        if kind == "heading_open":
            blocks.append(BlockNode("heading", level=int(token.tag[1:]), inline=_parse_inline(tokens[index + 1])))
            index += 3
        elif kind == "paragraph_open":
            blocks.append(BlockNode("paragraph", inline=_parse_inline(tokens[index + 1])))
            index += 3
        elif kind in {"bullet_list_open", "ordered_list_open"}:
            list_items, index = _parse_list(tokens, index, end)
            blocks.extend(list_items)
        elif kind == "blockquote_open":
            close = _find_matching_close(tokens, index, end)
            # 引用内容是 open/close 之间的兄弟 token，递归解析该区间。
            inner, _ = _parse_blocks(tokens, index + 1, close)
            blocks.append(BlockNode("quote", children=inner))
            index = close + 1
        elif kind in {"fence", "code_block"}:
            language = token.info.strip() if kind == "fence" else ""
            blocks.append(BlockNode("code", code=token.content.rstrip("\n"), language=language))
            index += 1
        elif kind == "html_block":
            blocks.append(BlockNode("code", code=token.content.rstrip("\n"), language="html"))
            index += 1
        elif kind == "hr":
            blocks.append(BlockNode("hr"))
            index += 1
        elif kind == "table_open":
            table_block, index = _parse_table(tokens, index, end)
            blocks.append(table_block)
        else:
            index += 1
    return blocks, end


def _parse_table(tokens, start: int, end: int) -> tuple[BlockNode, int]:  # noqa: ANN001
    """解析 table_open..table_close，返回 (BlockNode(table), table_close 后一索引)。"""
    table_close = _find_matching_close(tokens, start, end)
    rows: list[list[list[InlineNode]]] = []
    index = start + 1
    while index < table_close:
        token = tokens[index]
        if token.type == "tr_open":
            cells: list[list[InlineNode]] = []
            tr_close = _find_matching_close(tokens, index, table_close + 1)
            cursor = index + 1
            while cursor < tr_close:
                cell = tokens[cursor]
                if cell.type in {"th_open", "td_open"}:
                    cell_close = _find_matching_close(tokens, cursor, tr_close + 1)
                    # 单元格的 inline token 位于 open 与 close 之间。
                    for inner_index in range(cursor + 1, cell_close):
                        if tokens[inner_index].type == "inline":
                            cells.append(_parse_inline(tokens[inner_index]))
                    cursor = cell_close + 1
                else:
                    cursor += 1
            if cells:
                rows.append(cells)
            index = tr_close + 1
        else:
            index += 1
    return BlockNode("table", rows=rows), table_close + 1


def _parse_translation(markdown_text: str) -> list[BlockNode]:
    """解析译文 Markdown 为块级节点树（启用表格）。"""
    from markdown_it import MarkdownIt

    tokens = MarkdownIt("commonmark", {"html": True}).enable("table").parse(markdown_text)
    blocks, _ = _parse_blocks(tokens)
    return blocks


def _inline_to_json(nodes: list[InlineNode]) -> list[dict[str, object]]:
    """把内联节点序列化为 JSON 可写结构。"""
    result: list[dict[str, object]] = []
    for node in nodes:
        item: dict[str, object] = {"type": node.type}
        if node.text:
            item["text"] = node.text
        if node.href:
            item["href"] = node.href
        if node.children:
            item["children"] = _inline_to_json(node.children)
        result.append(item)
    return result


def _block_to_json(block: BlockNode) -> dict[str, object]:
    """把单个块节点序列化为 JSON 可写结构。"""
    item: dict[str, object] = {"type": block.type}
    if block.type == "heading":
        item["level"] = block.level
        item["inline"] = _inline_to_json(block.inline)
    elif block.type in {"paragraph", "list_item"}:
        item["ordered"] = block.ordered
        item["inline"] = _inline_to_json(block.inline)
        if block.children:
            item["children"] = [_block_to_json(child) for child in block.children]
    elif block.type == "code":
        item["language"] = block.language
        item["code"] = block.code
    elif block.type == "quote":
        item["children"] = [_block_to_json(child) for child in block.children]
    elif block.type == "table":
        item["rows"] = [[_inline_to_json(cell) for cell in row] for row in block.rows]
    return item


def build_translation_json(markdown_text: str) -> str:
    """把译文导出为结构化 JSON 字符串；hr 作为页分隔点切分 pages。"""
    blocks = _parse_translation(markdown_text)
    pages: list[list[dict[str, object]]] = [[]]
    for block in blocks:
        if block.type == "hr":
            pages.append([])
        else:
            pages[-1].append(_block_to_json(block))
    payload: dict[str, object] = {
        "format": "mineru.translation",
        "version": 1,
        "page_count": len(pages),
        "pages": [{"page": index + 1, "blocks": page_blocks} for index, page_blocks in enumerate(pages)],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _inline_plain(nodes: list[InlineNode]) -> str:
    """把内联节点展平为纯文本（用于表格等简单场景）。"""
    parts: list[str] = []
    for node in nodes:
        if node.type == "linebreak":
            parts.append("\n")
        elif node.text:
            parts.append(node.text)
        if node.children:
            parts.append(_inline_plain(node.children))
    return "".join(parts)


# ---------------------------------------------------------------------------
# DOCX（python-docx，环境已有）
# ---------------------------------------------------------------------------


def _docx_add_inline(paragraph, nodes: list[InlineNode], assets_dir: Path | None) -> None:  # noqa: ANN001
    """向内联段落写入 run，处理粗体/斜体/代码/链接/图片/换行。"""
    for node in nodes:
        if node.type == "text":
            # 保留软换行，避免整段挤成一行。
            parts = node.text.split("\n")
            for i, part in enumerate(parts):
                if i:
                    paragraph.add_run().add_break()
                if part:
                    paragraph.add_run(part)
        elif node.type == "linebreak":
            paragraph.add_run().add_break()
        elif node.type == "strong":
            run = paragraph.add_run(_inline_plain(node.children))
            run.bold = True
        elif node.type == "em":
            run = paragraph.add_run(_inline_plain(node.children))
            run.italic = True
        elif node.type == "code":
            run = paragraph.add_run(node.text)
            run.font.name = "Courier New"
        elif node.type == "math":
            paragraph.add_run(node.text)
        elif node.type == "link":
            paragraph.add_run(_inline_plain(node.children))
        elif node.type == "image":
            _docx_add_image(paragraph, node.href, assets_dir)


def _resolve_asset(relative: str, assets_dir: Path | None) -> Path | None:
    """在素材目录内解析相对图片路径；逃逸或不存在时返回 None（而非抛错中断导出）。"""
    if assets_dir is None:
        return None
    candidate = (assets_dir / relative).resolve()
    if assets_dir.resolve() not in candidate.parents and candidate != assets_dir.resolve():
        return None
    return candidate if candidate.is_file() else None


def _docx_add_image(paragraph, relative: str, assets_dir: Path | None) -> None:  # noqa: ANN001
    """向段落插入图片；图片缺失时以占位文字代替。"""
    path = _resolve_asset(relative, assets_dir)
    if path is None:
        paragraph.add_run(f"[图片缺失: {relative}]")
        return
    from docx.shared import Inches

    # 版心约 6 英寸，5.5 英寸保证大图不超出页面且保持等比。
    paragraph.add_run().add_picture(str(path), width=Inches(5.5))


def _docx_render_blocks(document, blocks: list[BlockNode], assets_dir: Path | None, depth: int = 0) -> None:  # noqa: ANN001
    """把块节点列表写入 DOCX document；depth 为列表缩进层级。"""
    from docx.shared import Pt

    for block in blocks:
        if block.type == "heading":
            document.add_heading(_inline_plain(block.inline), level=min(block.level, 4))
        elif block.type == "paragraph":
            paragraph = document.add_paragraph()
            _docx_add_inline(paragraph, block.inline, assets_dir)
        elif block.type == "list_item":
            paragraph = document.add_paragraph(style="List Bullet" if not block.ordered else "List Number")
            paragraph.paragraph_format.left_indent = Pt(18 * (depth + 1))
            _docx_add_inline(paragraph, block.inline, assets_dir)
            if block.children:
                _docx_render_blocks(document, block.children, assets_dir, depth + 1)
        elif block.type == "quote":
            for child in block.children:
                paragraph = document.add_paragraph(style="Intense Quote")
                if child.type == "paragraph":
                    _docx_add_inline(paragraph, child.inline, assets_dir)
        elif block.type == "code":
            paragraph = document.add_paragraph()
            run = paragraph.add_run(block.code)
            run.font.name = "Courier New"
            run.font.size = Pt(9)
        elif block.type == "hr":
            document.add_page_break()
        elif block.type == "table" and block.rows:
            table = document.add_table(rows=len(block.rows), cols=len(block.rows[0]))
            table.style = "Table Grid"
            for row_index, row in enumerate(block.rows):
                for col_index, cell_nodes in enumerate(row):
                    cell = table.cell(row_index, col_index)
                    cell.text = _inline_plain(cell_nodes)
                    if row_index == 0:
                        for run in cell.paragraphs[0].runs:
                            run.bold = True


def build_translation_docx(markdown_text: str, output_path: str | Path, *, assets_dir: str | Path | None = None) -> Path:
    """把译文导出为 DOCX 文件，返回写入路径。"""
    from docx import Document

    blocks = _parse_translation(markdown_text)
    document = Document()
    assets = Path(assets_dir) if assets_dir else None
    _docx_render_blocks(document, blocks, assets)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path


# ---------------------------------------------------------------------------
# LaTeX（自生成 .tex，正文按 $...$ 数学段保护，仅转义非数学文本）
# ---------------------------------------------------------------------------

_MATH_SPLIT_PATTERN = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape_text(text: str) -> str:
    """转义正文中的 LaTeX 特殊字符；$...$ 数学段原样保留。"""
    parts: list[str] = []
    for segment in _MATH_SPLIT_PATTERN.split(text):
        if not segment:
            continue
        if segment.startswith("$"):
            parts.append(segment)
            continue
        escaped = "".join(_LATEX_SPECIALS.get(char, char) for char in segment)
        parts.append(escaped)
    return "".join(parts)


def _latex_inline(nodes: list[InlineNode], assets_dir: Path | None) -> str:
    """渲染内联节点为 LaTeX 文本。"""
    parts: list[str] = []
    for node in nodes:
        if node.type == "text":
            parts.append(_latex_escape_text(node.text))
        elif node.type == "linebreak":
            parts.append("\\\\\n")
        elif node.type == "strong":
            parts.append(r"\textbf{" + _latex_inline(node.children, assets_dir) + "}")
        elif node.type == "em":
            parts.append(r"\textit{" + _latex_inline(node.children, assets_dir) + "}")
        elif node.type == "code":
            parts.append(r"\texttt{" + _latex_escape_text(node.text) + "}")
        elif node.type == "math":
            parts.append(node.text)
        elif node.type == "link":
            parts.append(_latex_inline(node.children, assets_dir))
        elif node.type == "image":
            parts.append(_latex_image(node.href, assets_dir))
    return "".join(parts)


def _latex_image(relative: str, assets_dir: Path | None) -> str:
    """生成图片 LaTeX 片段；缺失图片以占位文字代替。"""
    path = _resolve_asset(relative, assets_dir)
    if path is None:
        return _latex_escape_text(f"[图片缺失: {relative}]")
    return "\n\\begin{center}\n\\includegraphics[width=0.9\\linewidth]{" + path.name + "}\n\\end{center}\n"


def _latex_blocks(blocks: list[BlockNode], assets_dir: Path | None, depth: int = 0) -> str:
    """渲染块节点列表为 LaTeX 正文片段。"""
    lines: list[str] = []
    indent = "  " * depth
    for block in blocks:
        if block.type == "heading":
            # 用非编号标题，避免译文被自动编号；超出 4 级回落到 paragraph。
            mapped = ["section", "subsection", "subsubsection", "paragraph"][min(block.level - 1, 3)]
            lines.append(f"\n\\{mapped}*{{{_latex_inline(block.inline, assets_dir)}}}\n")
        elif block.type == "paragraph":
            lines.append("\n" + _latex_inline(block.inline, assets_dir) + "\n")
        elif block.type == "list_item":
            environment = "enumerate" if block.ordered else "itemize"
            marker = f"{indent}\\begin{{{environment}}}\n"
            lines.append(marker)
            lines.append(f"{indent}\\item {_latex_inline(block.inline, assets_dir)}\n")
            if block.children:
                lines.append(_latex_blocks(block.children, assets_dir, depth + 1))
            lines.append(f"{indent}\\end{{{environment}}}\n")
        elif block.type == "quote":
            lines.append("\n\\begin{quote}\n" + _latex_blocks(block.children, assets_dir, depth) + "\\end{quote}\n")
        elif block.type == "code":
            lines.append("\n\\begin{verbatim}\n" + block.code + "\n\\end{verbatim}\n")
        elif block.type == "hr":
            lines.append("\n\\newpage\n")
        elif block.type == "table" and block.rows:
            col_count = len(block.rows[0])
            lines.append("\n\\begin{longtable}{" + "l" * col_count + "}\n")
            for row_index, row in enumerate(block.rows):
                cells = " & ".join(_latex_inline(cell, assets_dir) for cell in row)
                lines.append(cells + " \\\\\n")
                if row_index == 0:
                    lines.append("\\hline\n")
            lines.append("\\end{longtable}\n")
    return "".join(lines)


_LATEX_PREAMBLE = """\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{fontspec}
\\usepackage{graphicx}
\\usepackage{longtable}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{hyperref}
\\setmainfont{Noto Serif CJK SC}
"""


def build_translation_latex(markdown_text: str, output_path: str | Path, *, assets_dir: str | Path | None = None) -> Path:
    """把译文导出为 .tex 文件（XeLaTeX 编译，含中文主字体）。"""
    blocks = _parse_translation(markdown_text)
    assets = Path(assets_dir) if assets_dir else None
    body = _latex_blocks(blocks, assets)
    content = _LATEX_PREAMBLE + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PDF（reportlab Platypus；STSong-Light 为内置中文 CID 字体，无需系统字体文件）
# ---------------------------------------------------------------------------


def _pdf_inline(nodes: list[InlineNode], assets_dir: Path | None) -> list:
    """把内联节点转换为 reportlab flowables（文本片段/图片）。"""
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Paragraph

    items: list = []
    text_parts: list[str] = []

    def flush() -> None:
        if not text_parts:
            return
        content = "".join(text_parts)
        text_parts.clear()
        if content.strip():
            items.append(Paragraph(content, _pdf_text_style()))

    for node in nodes:
        if node.type == "text":
            text_parts.append(_xml_escape(node.text).replace("\n", "<br/>"))
        elif node.type == "linebreak":
            text_parts.append("<br/>")
        elif node.type == "strong":
            text_parts.append("<b>" + _xml_escape(_inline_plain(node.children)) + "</b>")
        elif node.type == "em":
            text_parts.append("<i>" + _xml_escape(_inline_plain(node.children)) + "</i>")
        elif node.type == "code":
            text_parts.append('<font face="Courier">' + _xml_escape(node.text) + "</font>")
        elif node.type == "math":
            text_parts.append(_xml_escape(node.text))
        elif node.type == "link":
            text_parts.append(_xml_escape(_inline_plain(node.children)))
        elif node.type == "image":
            flush()
            path = _resolve_asset(node.href, assets_dir)
            if path is None:
                text_parts.append(_xml_escape(f"[图片缺失: {node.href}]"))
            else:
                image = _scaled_image(path, ImageReader(str(path)))
                items.append(image)
    flush()
    return items


def _scaled_image(path: Path, reader) -> object:  # noqa: ANN001
    """按原始像素等比缩放到版心（宽 ≤460pt、高 ≤640pt），避免大图溢出页面。"""
    from reportlab.platypus import Image

    width, height = reader.getSize()
    scale = min(460.0 / width, 640.0 / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def _xml_escape(text: str) -> str:
    """转义 reportlab Paragraph 标记语言中的 XML 特殊字符。"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _pdf_text_style():  # noqa: ANN202
    """构造正文段落样式；粗体/斜体通过 Paragraph 内联标记实现。"""
    from reportlab.lib.styles import ParagraphStyle

    return ParagraphStyle(
        "TranslationBody",
        fontName="STSong-Light",
        fontSize=11,
        leading=18,
        spaceAfter=6,
    )


def _pdf_blocks(blocks: list[BlockNode], assets_dir: Path | None, depth: int = 0) -> list:  # noqa: ANN201
    """把块节点列表转换为 reportlab flowables。"""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, Preformatted, Spacer, Table, TableStyle

    flowables: list = []
    for block in blocks:
        if block.type == "heading":
            size = {1: 20, 2: 16, 3: 14}.get(block.level, 12)
            style = ParagraphStyle(
                f"H{block.level}",
                fontName="STSong-Light",
                fontSize=size,
                leading=size * 1.4,
                spaceBefore=10,
                spaceAfter=8,
                textColor=colors.HexColor("#1f2329"),
            )
            flowables.append(Paragraph(_xml_escape(_inline_plain(block.inline)), style))
        elif block.type == "paragraph":
            flowables.extend(_pdf_inline(block.inline, assets_dir))
        elif block.type == "list_item":
            prefix = f"{depth + 1}. " if block.ordered else "• "
            left = 18 * (depth + 1)
            style = ParagraphStyle(
                "ListItem",
                fontName="STSong-Light",
                fontSize=11,
                leading=18,
                leftIndent=left,
                firstLineIndent=-12,
                spaceAfter=3,
            )
            flowables.append(Paragraph(prefix + _xml_escape(_inline_plain(block.inline)), style))
            if block.children:
                flowables.extend(_pdf_blocks(block.children, assets_dir, depth + 1))
        elif block.type == "quote":
            style = ParagraphStyle(
                "Quote",
                fontName="STSong-Light",
                fontSize=11,
                leading=18,
                leftIndent=18,
                textColor=colors.grey,
            )
            for child in block.children:
                if child.type == "paragraph":
                    flowables.append(Paragraph(_xml_escape(_inline_plain(child.inline)), style))
        elif block.type == "code":
            code_style = ParagraphStyle(
                "Code",
                fontName="Courier",
                fontSize=9,
                leading=12,
                backColor=colors.HexColor("#f6f8fa"),
            )
            flowables.append(Preformatted(block.code, code_style))
            flowables.append(Spacer(1, 6))
        elif block.type == "hr":
            flowables.append(PageBreak())
        elif block.type == "table" and block.rows:
            data = [[Paragraph(_xml_escape(_inline_plain(cell)), _pdf_text_style()) for cell in row] for row in block.rows]
            table = Table(data, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f6f8fa")),
                        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                    ]
                )
            )
            flowables.append(table)
            flowables.append(Spacer(1, 8))
    return flowables


def build_translation_pdf(markdown_text: str, output_path: str | Path, *, assets_dir: str | Path | None = None) -> Path:
    """把译文导出为 PDF 文件，返回写入路径。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate

    # 注册内置中文 CID 字体；重复注册会报错，故捕获忽略。
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except Exception:  # noqa: BLE001 - 已注册时直接复用
        pass

    blocks = _parse_translation(markdown_text)
    assets = Path(assets_dir) if assets_dir else None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    doc.build(_pdf_blocks(blocks, assets))
    return path


__all__ = [
    "AUTO_DETECT_LANGUAGE",
    "DEFAULT_PROVIDER",
    "DEFAULT_SOURCE_LANGUAGE",
    "DEFAULT_TARGET_LANGUAGE",
    "LANGUAGES",
    "PROVIDERS",
    "PAGE_SEPARATOR",
    "TranslationConfig",
    "TranslationError",
    "build_translation_docx",
    "build_translation_json",
    "build_translation_latex",
    "build_translation_pdf",
    "render_translation_html",
    "test_translation_connection",
    "translate_markdown",
]

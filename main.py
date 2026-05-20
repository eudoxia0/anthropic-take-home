import html
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic
import requests
from dotenv import load_dotenv

WIKI_HEADERS = {"User-Agent": "anthropic-take-home/0.1 (QA bot)"}


@dataclass(frozen=True)
class SearchResult:
    key: str
    title: str
    excerpt: str
    description: str


def search_wikipedia(query: str) -> list[SearchResult]:
    response = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 10},
        headers=WIKI_HEADERS,
    )
    response.raise_for_status()
    data = response.json()
    return [
        SearchResult(
            key=page["key"],
            title=page["title"],
            excerpt=page["excerpt"],
            description=page.get("description") or "",
        )
        for page in data["pages"]
    ]


def retrieve_page(key: str) -> str:
    response = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/page/" + key,
        headers=WIKI_HEADERS,
    )
    response.raise_for_status()
    data = response.json()
    return data["source"]


SYSTEM_PROMPT: str = """
Good morning Claude. Your task is to be the core of a question-answering
system. You have a couple tools at your disposal, which are you encouraged to
use:

- search_wikipedia: searches Wikipedia for the given query.
- retrieve_page: returns the source text of the given Wikipedia page.

Your answers should be grounded in the sources that you consult, which is to
say:

- Positive claims in the answer must be grounded in the sources.
- Don't ignore relevant information from the sources when composing your
  answer.

Rules:

- Answers should be succinct.
- You don't have to mention you got your data from Wikipedia.
- Where sources list multiple values for some data, only mention the most
  recent value.
"""

MODEL: str = "claude-opus-4-7"

TOOLS = [
    {
        "name": "search_wikipedia",
        "description": "Search Wikipedia for the given query and return a list of search results with keys, titles, excerpts, and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_page",
        "description": "Retrieve the full source text (in Wikitext) of a Wikipedia page by its key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The page key (from search results).",
                },
            },
            "required": ["key"],
        },
    },
]


@dataclass(frozen=True)
class SearchCall:
    query: str
    results: list[SearchResult]

    @property
    def name(self) -> str:
        return "search_wikipedia"

    @property
    def output_text(self) -> str:
        return json.dumps(
            [
                {
                    "key": r.key,
                    "title": r.title,
                    "excerpt": r.excerpt,
                    "description": r.description,
                }
                for r in self.results
            ]
        )


@dataclass(frozen=True)
class RetrievePageCall:
    key: str
    source: str

    @property
    def name(self) -> str:
        return "retrieve_page"

    @property
    def output_text(self) -> str:
        return self.source


ToolCall = SearchCall | RetrievePageCall


@dataclass
class QAResult:
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def _execute_tool(name: str, input: dict) -> tuple[ToolCall, str]:
    if name == "search_wikipedia":
        results = search_wikipedia(input["query"])
        call = SearchCall(query=input["query"], results=results)
        return call, call.output_text
    elif name == "retrieve_page":
        source = retrieve_page(input["key"])
        call = RetrievePageCall(key=input["key"], source=source)
        return call, call.output_text
    else:
        raise ValueError(f"Unknown tool: {name}")


def answer_question(question: str) -> QAResult:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    tool_calls: list[ToolCall] = []

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=TOOLS,  # type: ignore
        messages=messages,  # type: ignore
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                call, output = _execute_tool(block.name, block.input)
                tool_calls.append(call)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )

        messages.append({"role": "assistant", "content": response.content})  # type: ignore
        messages.append({"role": "user", "content": tool_results})  # type: ignore

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,  # type: ignore
            messages=messages,  # type: ignore
        )

    answer = "".join(block.text for block in response.content if block.type == "text")
    return QAResult(question=question, answer=answer, tool_calls=tool_calls)


def _esc(s: str) -> str:
    return html.escape(s)


def _write_log(result: QAResult) -> None:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = random.randint(0, 999999)
    filename = f"{timestamp}_{rand}.html"

    tool_calls_html = ""
    for tc in result.tool_calls:
        output_text = tc.output_text
        if isinstance(tc, SearchCall):
            input_str = f"query: {tc.query}"
            results_html = ""
            for sr in tc.results:
                results_html += (
                    f'<div class="search-result">'
                    f'<div class="search-result-header">'
                    f'<strong>{_esc(sr.title)}</strong>'
                    f'<span class="search-result-key">{_esc(sr.key)}</span>'
                    f'</div>'
                    f'<div class="search-result-desc">{_esc(sr.description)}</div>'
                    f'<pre class="search-result-excerpt">{_esc(sr.excerpt)}</pre>'
                    f'</div>'
                )
            output_html = results_html
        elif isinstance(tc, RetrievePageCall):
            input_str = f"key: {tc.key}"
            output_html = f'<pre class="tool-output">{_esc(output_text)}</pre>'
        tool_calls_html += (
            f'<div class="tool-call">'
            f'<strong>{_esc(tc.name)}</strong>'
            f'<pre class="tool-input">{_esc(input_str)}</pre>'
            f'<details><summary class="tool-output-toggle">Show output ({len(output_text)} chars)</summary>'
            f'{output_html}</details></div>'
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QA Log &mdash; {_esc(result.question[:80])}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 24px; background: #f3f4f6; color: #111827; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 24px; }}
    .section {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 16px 24px; margin-bottom: 16px; }}
    .section h2 {{ margin-top: 0; }}
    .question {{ font-size: 18px; }}
    .answer {{ white-space: pre-wrap; }}
    .tool-call {{ margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }}
    .tool-input {{ background: #f9fafb; padding: 8px; border-radius: 4px; margin: 6px 0; overflow-x: auto; font-size: 12px; }}
    .tool-output-toggle {{ cursor: pointer; color: #6b7280; font-size: 13px; }}
    .tool-output {{ background: #f9fafb; padding: 8px; border-radius: 4px; margin: 6px 0; overflow-x: auto; font-size: 11px; max-height: 400px; overflow-y: auto; }}
    .search-result {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; margin: 6px 0; background: #fff; }}
    .search-result-header {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }}
    .search-result-key {{ font-size: 12px; color: #9ca3af; font-family: monospace; }}
    .search-result-desc {{ font-size: 13px; color: #6b7280; margin-bottom: 6px; }}
    .search-result-excerpt {{ background: #f9fafb; padding: 8px; border-radius: 4px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto; }}
    .no-tools {{ color: #9ca3af; }}
</style>
</head>
<body>
<div class="container">
    <h1>QA Log</h1>
    <div class="subtitle">Generated {generated}</div>

    <div class="section">
        <h2>Question</h2>
        <div class="question">{_esc(result.question)}</div>
    </div>

    <div class="section">
        <h2>Tool Calls ({len(result.tool_calls)})</h2>
        {tool_calls_html if tool_calls_html else '<p class="no-tools">No tool calls made.</p>'}
    </div>

    <div class="section">
        <h2>Answer</h2>
        <div class="answer">{_esc(result.answer)}</div>
    </div>
</div>
</body>
</html>"""

    with open(os.path.join(log_dir, filename), "w") as f:
        f.write(page)


def main():
    load_dotenv()
    result = answer_question(sys.argv[1])
    _write_log(result)
    print(result.answer)


if __name__ == "__main__":
    main()

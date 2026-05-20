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
            description=page.get("description", ""),
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


def _write_log(result: QAResult) -> None:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = random.randint(0, 999999)
    filename = f"{timestamp}_{rand}.txt"
    sep = "\n" + "=" * 72 + "\n\n"
    parts = [f"[question]\n\n{result.question}"]
    for tc in result.tool_calls:
        if isinstance(tc, SearchCall):
            parts.append(f"[search_wikipedia] query={tc.query!r}\n\n{tc.output_text}")
        elif isinstance(tc, RetrievePageCall):
            parts.append(f"[retrieve_page] key={tc.key!r}\n\n{tc.output_text}")
    parts.append(f"[answer]\n\n{result.answer}")
    with open(os.path.join(log_dir, filename), "w") as f:
        f.write(sep.join(parts) + "\n")


def main():
    load_dotenv()
    result = answer_question(sys.argv[1])
    _write_log(result)
    print(result.answer)


if __name__ == "__main__":
    main()

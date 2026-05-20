import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic
import requests
from dotenv import load_dotenv

WIKI_HEADERS = {"User-Agent": "anthropic-take-home/0.1 (QA bot)"}


@dataclass(frozen=True)
class SearchResult:
    """
    Represents a search result from the MediaWiki API.
    """

    key: str
    title: str
    excerpt: str
    description: str


def search_wikipedia(query: str) -> list[SearchResult]:
    """
    Search Wikipedia for the given query, and return the list of search results.
    """
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
    """
    Retrieve the source text, in Wikitext, of the given Wikipedia page.
    """
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


def _execute_tool(name: str, input: dict, log: list[str]) -> str:
    if name == "search_wikipedia":
        results = search_wikipedia(input["query"])
        output = json.dumps(
            [
                {
                    "key": r.key,
                    "title": r.title,
                    "excerpt": r.excerpt,
                    "description": r.description,
                }
                for r in results
            ]
        )
        lines = [
            f"[search_wikipedia] query={input['query']!r}",
            "",
            output,
        ]
        log.append("\n".join(lines))
        return output
    elif name == "retrieve_page":
        result = retrieve_page(input["key"])
        lines = [
            f"[retrieve_page] key={input['key']!r}",
            "",
            result,
        ]
        log.append("\n".join(lines))
        return result
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


_LOG_SEPARATOR = "\n" + "=" * 72 + "\n\n"


def _write_log(log: list[str]) -> None:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = random.randint(0, 999999)
    filename = f"{timestamp}_{rand}.txt"
    with open(os.path.join(log_dir, filename), "w") as f:
        f.write(_LOG_SEPARATOR.join(log) + "\n")


def answer_question(question: str) -> str:
    """
    Entrypoint to the QA system: takes a question, invokes the model to
    synthesize an answer, optionally querying Wikipedia, and returns the final
    answer.
    """
    log: list[str] = [f"[question]\n\n{question}"]
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=TOOLS,  # type: ignore
        messages=messages,  # type: ignore
    )

    # Loop: handle tool use until we get a final text response.
    while response.stop_reason == "tool_use":
        # Collect all tool use blocks and execute them.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _execute_tool(block.name, block.input, log)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
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

    # Extract the final text.
    answer = "".join(block.text for block in response.content if block.type == "text")
    log.append(f"[answer]\n\n{answer}")
    _write_log(log)
    return answer


def main():
    load_dotenv()
    print(answer_question(sys.argv[1]))


if __name__ == "__main__":
    main()

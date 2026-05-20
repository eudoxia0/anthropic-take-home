import json
import sys
from dataclasses import dataclass

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
You are a question-answering system.

You have two tools at your disposal:

- search_wikipedia: searches Wikipedia for the given query.
- retrieve_page: returns the source text of the given Wikipedia page.
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


def _execute_tool(name: str, input: dict) -> str:
    if name == "search_wikipedia":
        results = search_wikipedia(input["query"])
        return json.dumps(
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
    elif name == "retrieve_page":
        return retrieve_page(input["key"])
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def answer_question(question: str) -> str:
    """
    Entrypoint to the QA system: takes a question, invokes the model to
    synthesize an answer, optionally querying Wikipedia, and returns the final
    answer.
    """
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
                result = _execute_tool(block.name, block.input)
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
    return answer


def main():
    load_dotenv()
    print(answer_question(sys.argv[1]))


if __name__ == "__main__":
    main()

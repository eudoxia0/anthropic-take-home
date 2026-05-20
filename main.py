import sys
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


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


def answer_question(question: str) -> str:
    """
    Entrypoint to the QA system: takes a question, invokes the model to
    synthesize an answer, optionally querying Wikipedia, and returns the final
    answer.
    """
    raise NotImplementedError()


def main():
    load_dotenv()
    answer_question(sys.argv[1])


if __name__ == "__main__":
    main()

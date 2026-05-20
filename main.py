from dataclasses import dataclass

import requests


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


def main():
    print("Hello from anthropic-take-home!")


if __name__ == "__main__":
    main()

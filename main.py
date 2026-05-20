from dataclasses import dataclass


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
    raise NotImplementedError()


def main():
    print("Hello from anthropic-take-home!")


if __name__ == "__main__":
    main()

# Anthropic Take Home

This is the prompt engineering take home!

## Setup

Install [uv]. Then, create a file `.env` with your anthropic API key:

```
ANTHROPIC_API_KEY=...
```

as in `.env.example`.

## Usage

To answer a single question, run:

```
uv run main.py $QUESTION
```

e.g.:

```
uv run main.py "Who is the current Secretary-General of NATO?"
```

The result will be rendered as an HTML file in the `logs/` directory.

To run the eval suite, run:

```
uv run eval.py
```

This will create an HTML file in the `reports/` directory.

[uv]: https://docs.astral.sh/uv/

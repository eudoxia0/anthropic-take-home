# Anthropic Take Home

This is the prompt engineering take home!

## Setup

Install [uv]. Then, create a file `.env` with your Anthropic API key:

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

This will create an HTML file in the `reports/` directory with the eval results.

[uv]: https://docs.astral.sh/uv/

## Rationale

This section describes the rationale for my design decisions.

### Architecture

I didn't spend too much time on this. System prompt + tool use. A more complex
setup (e.g. with a reviewer agent who can approve/reject answers) would be
harder to test because it would require more complex questions.

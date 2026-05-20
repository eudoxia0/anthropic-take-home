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

### Qualitative Evaluation

In my experience building LLM harnesses, qualitative evaluation is where most of
the wins are. This means being able to read transcripts (question + tool calls
and outputs + answer) comfortably. Reading a multi-line string in a JSON
document in a log line in a terminal is uncomfortable. So I spent some time
getting Claude Code to build a system to render results to nicely-formatted
HTML, so I can read the transcript more effectively.

This was valuable, in particular at seeing where Claude would and would not use
tools.

### Prompt Engineering

My approach to writing the prompt was: start with something very minimal,
iterate based on results. Short prompts are good because the longer the prompt,
the harder it is to coordinate it and ensure it's not accidentally repetitive,
contradictory etc. I try to act like I'm addressing a smart, literate human, who
wants clear, unambiguous instructions, and not a machine with cheat codes,
i.e. no "you're a 150IQ genius and if you fail my family dies".

### Dimensions of Quality

### Evaluation Lessons

### Key Iterations

### How I Would Extend This

### AI Usage

### Time Taken

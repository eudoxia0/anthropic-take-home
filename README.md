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

### Evaluation Architecture

This was also simple: an LLM evaluates the transcript along the different
dimensions of quality. I picked Sonnet on the idea that judging is less
cognitively demanding than evaluation.

Claude Code wrote the eval harness, in particular it implemented a 1-5 score
scale, which I accepted. Part of me thinks this is too much resolution, since in
practice all answers were graded 5/5 or 4/5 when the judge had a nit. But, in
the context of regression testing, having extra room to go down might be useful,
so the judges can report severe degradations.

### Dimensions of Quality

The first dimension that came to mind was **accuracy**, which I broke down into
false positives, false negatives, and consistency:

- **No false positives:** positive claims in the answer must be grounded in
  sources.
- **No false negatives:** answers should not ignore key facts in the sources.
- **Consistency:** the answer must not contradict the sources, i.e., an answer
can't assert `P` if the sources say `not P`, and vice-versa.

After asking Claude for other dimensions I may have overlooked, I got the idea
to also address **relevance**, i.e. whether the answer actually answers the
question.

### Evaluation Lessons



### Key Iterations

### How I Would Extend This

### AI Usage

### Time Taken

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
harder to test because it would require more complex questions. A tiny design
decision that was useful was pulling the Wikitext source of the page rather than
the rendered HTML, since this is more compact.

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

Beyond that, I explained the task to Claude, and showed it the prompt, and asked
it how it would improve it, on the theory that the models are introspective
enough that they can describe how they want to be prompted, and what they think
is ambiguous in the prompt.

I like explaining why in the prompts, partly because you would explain the why
to a human, and partly because I think by explaining the how and the why, the
model can understand the how better.

### Evaluation Architecture

This was also simple: an LLM evaluates the transcript along the different
dimensions of quality. I picked Sonnet on the idea that judging is less
cognitively demanding than evaluation.

Claude Code wrote the eval harness, in particular it implemented a 1-5 score
scale, which I accepted. Part of me thinks this is too much resolution, since in
practice all answers were graded 5/5 or 4/5 when the judge had a tiny
comment. But, in the context of regression testing, having extra room to go down
might be useful, so the judges can report severe degradations.

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

Qualitatively, the main lesson came from scanning the transcripts, seeing where
Claude was not using the tools, or relying on unexpected output. For example
sometimes Claude would make a query, and base an answer from the excerpt of a
Wikipedia search result: I altered the prompt so that it would always retrieve
the page.

Quantitatively, the scores were mostly helpful in that anything lower than 5/5
made me review the transcript, and sometimes I'd alter the prompt somewhat in
response to the judge's output.

### Key Iterations

The architecture and evaluation frameworks mostly didn't change. The main thing
that changed was the QA prompt.

The initial version of the prompt was very minimal, just to get the system
going. I then wrote a longer version, with more rules, and found it mostly
worked, but the model frequently answered from memory, and ignored instructions
to use its tools. After consulting with Claude, I made the prompt a bit
stricter, less ambiguous, and explained the rationale for the rules. This
version of the prompt worked: it pulled data from Wikipedia and generated
answers grounded in sources.

But, sometimes the judges would have little nitpicks. These effectively worked
like false alarms. I asked the model to be more succinct, which reduced the
surface area for judges to score less than 5/5.

### How I Would Extend This

It's hard to quantitatively evaluate models with a small _n_ sample, and it's
hard to increase _n_ since it's hard to generate questions which are:

1. Tractable enough that they can be answered from Wikipedia alone, and
2. Complex enough that the models can get them wrong so we can examine failure
   modes, and
3. Are not pathological.

So, I would try to source some more high-quality questions, but _n_ is unlikely
to be very high.

I would probably narrow the scoring criteria to: perfect, flawed, wrong. This is
more qualitative, and less quantitative. A `flawed` value means the answer and
judge comments are worth looking at, and maybe updating the judge's prompt so
it's more focused. A `wrong` value might be a signal to stop model rollout if
enough questions are wrong.

I would make a few tiny changes to the eval framework:

- Have each eval criterion answered by a separate LLM call, to make the output
  more focused.
- Structure the output so the reasoning is before the score. This is just
  generally a good practice.

### AI Usage

Claude Code transcripts are in the `claude-code/` directory.

I used Claude Code both to implement valuable quality-of-life features (like
HTML rendering) in less time, and to make focused, narrow changes
(e.g. implement the body of a function where I've written the signature).

### Time Taken

I read the README a few times, then went out to get lunch while thinking about
it in the back of my head. That was about 30m. I allocated 2h to write all the
code end-to-end. I had mostly finished by 1h40m and used the last 20m to
experiment some more with the prompt. Writing the rationale took about 40m.

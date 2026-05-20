import html
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

from main import QAResult, RetrievePageCall, SearchCall, answer_question

JUDGE_MODEL = "claude-sonnet-4-6"

EVAL_QUESTIONS = [
    # Single source.
    "Who is the current Secretary-General of NATO?",
    "What was the most recent film to win the Palme d'Or?",
    "What is the population of Tokyo?",
    "When was the Great Wall of China built?",
    "How many moons does Saturn currently have confirmed?",
    # Multiple source.
    "What is the population of the city where the 2028 Summer Olympics will be held?",
    "What is the capital of the country whose football team won the most recent FIFA World Cup?",
    "What is the elevation of the highest peak in the country that currently chairs the G20?",
]

JUDGE_SYSTEM_PROMPT_TEMPLATE = """\
Today's date is TODAY.

You are an evaluation judge for a question-answering system that uses Wikipedia \
as its source. You will be given:

1. The question that was asked.
2. The tool calls the system made (Wikipedia searches and page retrievals) and their results.
3. The final answer the system produced.

You must evaluate the answer on four criteria, each scored from 1 to 5:

1. **No False Positives**: Positive claims in the answer must be grounded in the \
sources retrieved. Score 1 if the answer contains fabricated claims not supported by \
sources; score 5 if every claim is well-grounded.

2. **No False Negatives**: The answer should not ignore facts in the sources that \
are directly relevant and would materially change the answer. Score 1 if critical \
source information is ignored; score 5 if all relevant source information is used.

3. **Consistency**: The answer must not contradict the sources. Score 1 if the answer \
directly contradicts source material; score 5 if fully consistent.

4. **Relevance**: The answer must actually address the question asked. Score 1 if \
the answer is off-topic; score 5 if it directly and completely answers the question.

Respond with a JSON object (no markdown fences) with exactly this structure:
{
  "no_false_positives": {"score": <1-5>, "reasoning": "<brief explanation>"},
  "no_false_negatives": {"score": <1-5>, "reasoning": "<brief explanation>"},
  "consistency": {"score": <1-5>, "reasoning": "<brief explanation>"},
  "relevance": {"score": <1-5>, "reasoning": "<brief explanation>"}
}
"""


@dataclass
class Judgment:
    no_false_positives: int
    no_false_negatives: int
    consistency: int
    relevance: int
    raw: dict


def judge(qa: QAResult) -> Judgment:
    tool_call_text = ""
    for tc in qa.tool_calls:
        output = tc.output_text
        if isinstance(tc, SearchCall):
            tool_call_text += (
                f"Tool: search_wikipedia\nQuery: {tc.query}\nOutput:\n{output}\n\n"
            )
        elif isinstance(tc, RetrievePageCall):
            tool_call_text += (
                f"Tool: retrieve_page\nKey: {tc.key}\nOutput:\n{output}\n\n"
            )

    user_prompt = (
        f"## Question\n\n{qa.question}\n\n"
        f"## Tool Calls and Results\n\n{tool_call_text}\n"
        f"## Final Answer\n\n{qa.answer}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT_TEMPLATE.replace(
            "TODAY", datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ),
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    data = json.loads(text)

    return Judgment(
        no_false_positives=data["no_false_positives"]["score"],
        no_false_negatives=data["no_false_negatives"]["score"],
        consistency=data["consistency"]["score"],
        relevance=data["relevance"]["score"],
        raw=data,
    )


@dataclass
class EvalResult:
    qa: QAResult
    judgment: Judgment


def _esc(s: str) -> str:
    return html.escape(s)


def generate_html(results: list[EvalResult]) -> str:
    rows = ""
    for i, r in enumerate(results):
        avg = (
            r.judgment.no_false_positives
            + r.judgment.no_false_negatives
            + r.judgment.consistency
            + r.judgment.relevance
        ) / 4.0

        def _badge(score: int) -> str:
            if score >= 4:
                cls = "badge-good"
            elif score >= 3:
                cls = "badge-ok"
            else:
                cls = "badge-bad"
            return f'<span class="badge {cls}">{score}/5</span>'

        tool_calls_html = ""
        for tc in r.qa.tool_calls:
            output_text = tc.output_text
            if isinstance(tc, SearchCall):
                input_str = f"query: {tc.query}"
                results_html = ""
                for sr in tc.results:
                    results_html += (
                        f'<div class="search-result">'
                        f'<div class="search-result-header">'
                        f"<strong>{_esc(sr.title)}</strong>"
                        f'<span class="search-result-key">{_esc(sr.key)}</span>'
                        f"</div>"
                        f'<div class="search-result-desc">{_esc(sr.description)}</div>'
                        f'<pre class="search-result-excerpt">{_esc(sr.excerpt)}</pre>'
                        f"</div>"
                    )
                output_html = results_html
            elif isinstance(tc, RetrievePageCall):
                input_str = f"key: {tc.key}"
                output_html = f'<pre class="tool-output">{_esc(output_text)}</pre>'
            tool_calls_html += (
                f'<div class="tool-call">'
                f"<strong>{_esc(tc.name)}</strong>"
                f'<pre class="tool-input">{_esc(input_str)}</pre>'
                f'<details><summary class="tool-output-toggle">Show output ({len(output_text)} chars)</summary>'
                f"{output_html}</details></div>"
            )

        judge_html = ""
        for key, label in [
            ("no_false_positives", "No False Positives"),
            ("no_false_negatives", "No False Negatives"),
            ("consistency", "Consistency"),
            ("relevance", "Relevance"),
        ]:
            entry = r.judgment.raw[key]
            judge_html += (
                f'<div class="judge-criterion">'
                f"<strong>{label}:</strong> {_badge(entry['score'])} "
                f'<span class="judge-reasoning">{_esc(entry["reasoning"])}</span>'
                f"</div>"
            )

        rows += f"""
        <tr class="clickable" data-idx="{i}">
            <td class="cell cell-num">{i + 1}</td>
            <td class="cell">{_esc(r.qa.question)}</td>
            <td class="cell cell-answer"><div class="answer-preview">{_esc(r.qa.answer)}</div></td>
            <td class="cell cell-score">{_badge(r.judgment.no_false_positives)}</td>
            <td class="cell cell-score">{_badge(r.judgment.no_false_negatives)}</td>
            <td class="cell cell-score">{_badge(r.judgment.consistency)}</td>
            <td class="cell cell-score">{_badge(r.judgment.relevance)}</td>
            <td class="cell cell-score cell-avg">{avg:.1f}</td>
        </tr>
        <tr id="detail-{i}" class="detail-row">
            <td colspan="8" class="detail-cell">
                <h3 class="detail-heading">Answer</h3>
                <div class="detail-answer">{_esc(r.qa.answer)}</div>
                <h3>Tool Calls ({len(r.qa.tool_calls)})</h3>
                {tool_calls_html if tool_calls_html else '<p class="no-tools">No tool calls made.</p>'}
                <h3>Judge Output</h3>
                {judge_html}
            </td>
        </tr>"""

    n = len(results)
    avg_fp = sum(r.judgment.no_false_positives for r in results) / n if n else 0
    avg_fn = sum(r.judgment.no_false_negatives for r in results) / n if n else 0
    avg_con = sum(r.judgment.consistency for r in results) / n if n else 0
    avg_rel = sum(r.judgment.relevance for r in results) / n if n else 0
    avg_all = (avg_fp + avg_fn + avg_con + avg_rel) / 4.0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QA Eval Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 24px; background: #f3f4f6; color: #111827; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 24px; }}
    .summary {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
    .summary-card {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; min-width: 140px; }}
    .summary-card .label {{ font-size: 13px; color: #6b7280; margin-bottom: 4px; }}
    .summary-card .value {{ font-size: 28px; font-weight: 700; }}
    table {{ width: 100%; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-collapse: collapse; }}
    th {{ background: #f9fafb; padding: 12px; text-align: left; font-size: 13px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 2px solid #e5e7eb; }}
    .clickable {{ cursor: pointer; border-bottom: 1px solid #e5e7eb; }}
    .clickable:hover {{ background: #f9fafb; }}
    .cell {{ padding: 12px; vertical-align: top; }}
    .cell-num {{ font-weight: 500; }}
    .cell-answer {{ max-width: 400px; }}
    .cell-score {{ text-align: center; }}
    .cell-avg {{ font-weight: bold; }}
    .answer-preview {{ max-height: 100px; overflow: hidden; text-overflow: ellipsis; }}
    .badge {{ color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
    .badge-good {{ background: #22c55e; }}
    .badge-ok {{ background: #eab308; }}
    .badge-bad {{ background: #ef4444; }}
    .detail-row {{ display: none; background: #f9fafb; }}
    .detail-cell {{ padding: 16px; }}
    .detail-heading {{ margin-top: 0; }}
    .detail-answer {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin-bottom: 16px; white-space: pre-wrap; }}
    .tool-call {{ margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }}
    .tool-input {{ background: #f9fafb; padding: 8px; border-radius: 4px; margin: 6px 0; overflow-x: auto; font-size: 12px; }}
    .tool-output-toggle {{ cursor: pointer; color: #6b7280; font-size: 13px; }}
    .tool-output {{ background: #f9fafb; padding: 8px; border-radius: 4px; margin: 6px 0; overflow-x: auto; font-size: 11px; max-height: 400px; overflow-y: auto; }}
    .search-result {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; margin: 6px 0; background: #fff; }}
    .search-result-header {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }}
    .search-result-key {{ font-size: 12px; color: #9ca3af; font-family: monospace; }}
    .search-result-desc {{ font-size: 13px; color: #6b7280; margin-bottom: 6px; }}
    .search-result-excerpt {{ background: #f9fafb; padding: 8px; border-radius: 4px; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto; }}
    .judge-criterion {{ margin-bottom: 8px; }}
    .judge-reasoning {{ color: #6b7280; font-size: 13px; }}
    .no-tools {{ color: #9ca3af; }}
</style>
</head>
<body>
<div class="container">
    <h1>QA Eval Report</h1>
    <div class="subtitle">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")} &mdash; {n} questions</div>

    <div class="summary">
        <div class="summary-card"><div class="label">No False Positives</div><div class="value">{avg_fp:.2f}</div></div>
        <div class="summary-card"><div class="label">No False Negatives</div><div class="value">{avg_fn:.2f}</div></div>
        <div class="summary-card"><div class="label">Consistency</div><div class="value">{avg_con:.2f}</div></div>
        <div class="summary-card"><div class="label">Relevance</div><div class="value">{avg_rel:.2f}</div></div>
        <div class="summary-card"><div class="label">Overall</div><div class="value">{avg_all:.2f}</div></div>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Question</th>
                <th>Answer</th>
                <th>No FP</th>
                <th>No FN</th>
                <th>Consist.</th>
                <th>Relev.</th>
                <th>Avg</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</div>
<script>
document.querySelectorAll('tr.clickable').forEach(function(row) {{
    row.addEventListener('click', function() {{
        var detail = document.getElementById('detail-' + this.dataset.idx);
        detail.style.display = detail.style.display === 'none' ? 'table-row' : 'none';
    }});
}});
</script>
</body>
</html>"""


def main():
    load_dotenv()

    results: list[EvalResult] = []
    for i, question in enumerate(EVAL_QUESTIONS):
        print(f"[{i + 1}/{len(EVAL_QUESTIONS)}] {question}")

        print("  Answering...")
        qa = answer_question(question)
        print(f"  Answer: {qa.answer[:100]}...")

        print("  Judging...")
        j = judge(qa)
        print(
            f"  Scores: FP={j.no_false_positives} FN={j.no_false_negatives} "
            f"C={j.consistency} R={j.relevance}"
        )

        results.append(EvalResult(qa=qa, judgment=j))

    # Write HTML report
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = random.randint(0, 999999)
    filename = f"eval_{timestamp}_{rand}.html"
    filepath = os.path.join(report_dir, filename)

    html_content = generate_html(results)
    with open(filepath, "w") as f:
        f.write(html_content)

    print(f"\nReport written to {filepath}")


if __name__ == "__main__":
    main()

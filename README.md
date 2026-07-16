# MES Intelligent Agentic AI System

## What this project does

This is a multi-agent manufacturing-defect analysis demo. A Streamlit UI lets a user pick a
defect type from a dropdown; a **Supervisor** AI agent (Claude, via the
[Strands Agents SDK](https://github.com/strands-agents)) then orchestrates five specialist
subagents — **Monitor, Analyzer, Planner, Verifier, Executor** — that query a synthetic MES
(Manufacturing Execution System) SQLite database, identify likely root causes for the selected
defect, and produce a structured PDF report plus a simulated ("dry-run") email notification.

All data in this project is synthetic. This is a learning / proof-of-concept project, not
production software.

## How to install requirements

From the project root, with a Python virtual environment created and activated:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `streamlit`, `pandas`, `numpy`, `plotly`, `python-dotenv`,
`reportlab` (PDF generation), `Faker` and `SQLAlchemy` (used by the separate synthetic-data
generator), the agent framework (`strands-agents[anthropic]`, `strands-agents-tools`,
`anthropic`), `boto3[crt]` (used by the email-notification tool), and optional Bedrock
AgentCore packages.


## Required environment variables

All configuration is read from a `.env` file in the project root at import time. **Never commit
real secret values** — only the variable names and their purpose are listed below.

| Variable | Required? | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** — the app raises an error at startup without it | Authenticates with the Claude/Anthropic API |
| `MES_MODEL_ID` | No (defaults to `claude-sonnet-4-6`) | Which Claude model all agents use |
| `MES_MAX_TOKENS` | No (defaults to `8296`) | Max output tokens per model call |
| `MES_TEMPERATURE` | No (defaults to `0.2`) | Model sampling temperature |
| `MES_LOG_LEVEL` | No (defaults to `INFO`) | Logging verbosity (`DEBUG` surfaces rate-limit/retry detail) |
| `MES_LOG_FORMAT` | No (has a default format string) | Log line format |
| `MES_DB_PATH` | No (defaults to `mes.db` in the working directory) | Path to the SQLite MES database |
| `AWS_REGION` | No (defaults to `us-west-2`) | AWS region for the SES email client |
| `MES_SENDER_EMAIL` | No (has a placeholder default) | "From" address for the notification email |
| `MES_RECIPIENT_EMAIL` | No (has a placeholder default) | "To" address for the notification email |
| `MES_BASE_URL` | No (has a default demo URL) | Base URL used to build the PDF link in emails |
| `MES_API_TIMEOUT` | No (defaults to `40` seconds) | HTTP timeout for the Anthropic client |
| `MES_API_RETRIES` | No (defaults to `2`) | Max retries for the Anthropic client |
| `MES_EMAIL_DRY_RUN` | No (defaults to `true`) | When true, the notification step logs/returns a message instead of actually sending email |


## How to run it

**Option A — automated bootstrap:**

```bash
python startup.py
```

This creates a `.venv` if missing, creates/repairs `.env` (prompting interactively for
`ANTHROPIC_API_KEY` if it's missing or malformed), installs `requirements.txt`, and launches
Streamlit for you.

**Option B — manual:**

```bash
# from the project root, in Git Bash
source .venv/Scripts/activate        # Windows venv path is Scripts, not bin
python -m streamlit run app.py       # serves at http://localhost:8501
```

Ensure `.env` exists with at least `ANTHROPIC_API_KEY` set before starting either way. `app.py`
is the Streamlit entry point: it renders the sidebar (look-back period, analysis-scope
checkboxes, defect-type dropdown) and a single **"Run Analysis"** button. That button is
intentionally the *only* way to trigger an analysis — Strands agents are not safely re-entrant,
so the app guards analysis start with one-shot session-state flags rather than any
auto-triggering logic.

## How the agents work together

All agent logic lives in `strands_agent.py` (`MESAgentManager` class). Each of the five
specialist agents is a separate `Agent` instance with its own system prompt and tool set:

- **Monitor** — answers *"what happened?"* Captures OEE drops, downtime/stoppage events,
  changeover times, and shift/operator context, and pulls historical data for comparison.
- **Analyzer** — answers *"why did it happen?"* Performs root-cause and correlation analysis
  (e.g. downtime vs. shift/operator/product), rating its own certainty as HIGH / MEDIUM / LOW.
- **Planner** — answers *"what should we do about it?"* Turns the analysis into a prioritized
  action plan (immediate / short-term / long-term) with resources, timelines, and KPIs.
- **Verifier** — answers *"can we trust this?"* Checks findings against baselines and flags
  whether a human needs to review the result (e.g. very low OEE, long downtime). It does not
  send any notifications itself.
- **Executor** — answers *"what do we do now?"* Converts the plan into concrete MES actions and
  drafts/sends the notification email (dry-run by default) with a link to the PDF report.

The **Supervisor** agent orchestrates all five, exposing each of them as a callable tool
(`call_monitor_agent`, `call_analyzer_agent`, `call_planner_agent`, `call_verifier_agent`,
`call_executor_agent`). Its system prompt prescribes a fixed sequence — Monitor → Analyzer →
Planner → Verifier → Executor — and, once all five have responded, compiles one final report
with exactly eight mandated sections: Defect Occurrence Summary, Maintenance Correlation
Findings, Root Cause Hypotheses, Data Reliability Flags, Gaps/Missing Data, Action Plan,
Verification Outcome, and Notification Status.

Every agent's system prompt ends with a shared "output rules" block designed to stop
fabrication: no invented totals/percentages/confidence numbers, no comparisons against
benchmarks that didn't come from a tool result, and every figure must be traceable to an actual
subagent or database query result.

## Where the database is located

`mes.db` — a SQLite file at the project root (its path is configurable via `MES_DB_PATH`, which
defaults to `mes.db` in the current working directory). It holds 14 tables of synthetic
manufacturing data (`Machines`, `WorkOrders`, `Downtimes`, `Defects`, `QualityControl`,
`OEEMetrics`, etc.), generated by a separate nested project
(`industrial-data-store-simulation-chatbot/app_factory/data_generator/`).

This database is meant to be treated as **read-only** by this application — the code path never
issues `INSERT`/`UPDATE`/`DELETE` against it, and all queries are parameterized and restricted
to an allowlist of tables/predefined queries (`_execute_safe_query`, `_validate_table_name`) so
the model can never construct or run arbitrary SQL. That said, the SQLite connection itself is
opened in normal read/write mode — the read-only guarantee is a coding convention enforced by
the query layer, not something the database connection itself enforces.

## How PDF reports are generated

PDF generation is not something the Supervisor calls as a tool mid-analysis — it happens once,
after the Supervisor finishes producing its full markdown-style final report text.
`MESAgentManager.run_defect_analysis()` then passes that returned text to a module-level
function, `render_markdown_report_pdf(markdown_text, filename)` (`strands_agent.py`), which:

1. Requires `reportlab` to be installed (raises a clear error if not).
2. Converts the markdown-ish text into ReportLab flowables — headings, bullet lists, simple
   tables — via a small helper (`_markdown_to_flowables` / `_md_inline`).
3. Builds the PDF with `reportlab.platypus.SimpleDocTemplate` on A4 pages, with a title page
   header ("Manufacturing Execution System Analysis Report" + a generation timestamp).
4. Writes the file to the `reports/` folder (created automatically beside `strands_agent.py` if
   it doesn't exist) as `<filename>.pdf`, where `filename` follows the pattern
   `MES_Final_Report_YYYYMMDD_HHMMSS` (timestamped, not tagged with the defect type). The
   filename is also embedded in the Executor agent's dry-run email as the report link.

⚠️ *Worth flagging:* in the current source, `run_defect_analysis()` calls
`render_markdown_report_pdf(...)` with the same report text and the same filename **twice in a
row** (once around line 1795, again around line 1825) — the second call simply re-renders and
overwrites the same PDF the first call already produced. This looks like redundant/duplicated
code rather than an intentional retry or draft/final step, but I have not changed it since this
task is documentation-only.

## Current limitations

- **Synthetic data only.** All MES data is generated, not from a real factory — outputs are for
  demonstration purposes only, not real operational decisions.
- **Latency.** A full analysis run calls the Claude model multiple times in sequence (once per
  specialist agent, plus the Supervisor's own orchestration and final-report synthesis), so a
  single run can take several minutes rather than returning instantly.
- **Report validation is still shallow.** The Verifier agent's confidence-scoring tool
  (`validate_findings`) is a stub that returns a hardcoded confidence score rather than
  performing real schema/provenance-based validation — the "Verification Outcome" section of a
  generated report should be treated as a placeholder today, not a rigorously validated
  confidence figure. Further work on real MES report validation is needed before these reports
  should inform real decisions.
- **Redundant PDF rendering.** As noted above, the final report is rendered to PDF twice per
  run with identical inputs — wasted work, though not currently harmful beyond that.
- **No real email is sent by default** (`MES_EMAIL_DRY_RUN=true`) — the notification step only
  simulates sending and logs what would have gone out.
- **Single-trigger design.** The UI deliberately allows only one way to start an analysis (the
  "Run Analysis" button) because the underlying agents are not safely re-entrant; there is no
  way to queue or run multiple analyses concurrently from the UI.
- **No computer-vision defect detection yet.** The project's roadmap describes replacing the
  manual defect-type dropdown with an image-based detection step (first a fake detection
  dictionary, later a real model), but this is not present in the current code — the dropdown
  is still the only selection method.
- ⚠️ *Not verified:* whether `startup.py` and `mes_config.sh` (which references AWS
  Bedrock-style configuration) are both actively maintained, or whether one is a legacy/
  experimental alternate path; and whether the SQLite driver used in this environment
  enforces the read-only convention any harder than what's described above.

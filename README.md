# MES Intelligent Agentic AI System

A multi-agent manufacturing-defect analysis demo: a Streamlit UI lets you pick a defect type,
and a supervisor AI agent orchestrates five specialist agents (Monitor, Analyzer, Planner,
Verifier, Executor) to investigate a synthetic MES (Manufacturing Execution System) database,
identify likely root causes, and produce a PDF report plus a simulated ("dry-run") email
notification.

> All data used by this project is synthetic. This is a learning / proof-of-concept project,
> not production software.

## Project purpose

To demonstrate a closed-loop, agentic approach to manufacturing quality investigation: instead
of a human manually querying MES tables after a defect spike, a team of cooperating AI agents
does the querying, correlation, root-cause reasoning, action planning, and reporting, using
Claude (via the [Strands Agents SDK](https://github.com/strands-agents)) as the reasoning
engine over a real (synthetic) SQLite dataset.

## Main features

- Streamlit dashboard for selecting a defect type and analysis scope (look-back period; OEE,
  downtime, changeover, and maintenance correlation toggles).
- A five-agent pipeline (see below) that investigates the selected defect end-to-end.
- Structured, multi-section markdown report compiled by the supervisor agent, covering defect
  occurrence, root-cause hypotheses, data reliability flags, gaps in the data, an action plan,
  and a verification outcome.
- Automatic PDF report generation (via ReportLab) for each analysis run.
- A dry-run email notification step (no real email is sent by default).
- All SQL access is parameterized and restricted to an allowlist of tables/queries — no raw
  free-form SQL is ever executed against the database by the model.

## High-level architecture

```
Streamlit UI (app.py)
        │  "Run Analysis" button
        ▼
MESAgentManager (strands_agent.py)
        │
        ▼
  Supervisor agent  ───────────────────────────────────────────┐
        │  calls each subagent as a tool, in sequence           │
        ▼                                                       │
  Monitor → Analyzer → Planner → Verifier → Executor            │
   (what      (why       (what to   (can we    (do it —         │
   happened)  did it     do about   trust      MES actions +    │
              happen)    it)        this?)     dry-run email)   │
        │                                                       │
        └──────────────► compiled markdown report ◄─────────────┘
                                  │
                                  ▼
                    render_markdown_report_pdf()
                                  │
                                  ▼
                         reports/*.pdf saved
```

An architecture diagram image is also included in the repository root:
`architecture_process_optimization.png`.

*(⚠️ Not verified in detail: whether that image reflects the exact current code path, since I
have not rendered/inspected the image contents myself — only confirmed the file exists.)*

## Supervisor and specialist agents

Each agent is a separate Strands `Agent` instance with its own system prompt and tool set,
defined in `strands_agent.py`.

- **Supervisor** — the orchestrator. Exposes each specialist agent as a callable tool
  (`call_monitor_agent`, `call_analyzer_agent`, `call_planner_agent`, `call_verifier_agent`,
  `call_executor_agent`) and follows a fixed sequence: Monitor → Analyzer → Planner → Verifier →
  Executor. After all five have run, it compiles one structured report with mandated sections
  (Defect Occurrence Summary, Maintenance Correlation, Root Cause Hypotheses, Data Reliability
  Flags, Gaps/Missing Data, Action Plan, Verification Outcome, Notification Status).
- **Monitor** — answers "what happened?" Pulls OEE drops, downtime/stoppage events, changeover
  times, shift/operator context, and historical comparisons from the database.
- **Analyzer** — answers "why did it happen?" Performs root-cause and correlation reasoning
  (e.g. downtime vs. shift/operator/product), and rates its own certainty as HIGH / MEDIUM / LOW.
- **Planner** — answers "what should we do about it?" Converts the analysis into a prioritized
  action plan (immediate / short-term / long-term), with resources, timelines, and KPIs.
- **Verifier** — answers "can we trust this?" Checks findings against baselines and decides
  whether a human needs to review the result (e.g. very low OEE, long downtime). Does not send
  any notifications itself.
- **Executor** — answers "what do we do now?" Turns the plan into concrete MES actions and
  drafts/sends the summary notification (dry-run by default) with a link to the generated PDF.

Every agent's system prompt ends with a shared "output rules" block whose intent is to prevent
fabricated numbers or invented confidence levels — reported figures are expected to trace back
to an actual database query result, not to be estimated by the model.

## Installation instructions

1. Clone this repository.
2. Make sure you have a compatible Python installed (see **Required Python version** below).
3. Create and activate a virtual environment, or let `startup.py` do it for you (see below).
4. Install dependencies from `requirements.txt`.
5. Provide an `.env` file at the project root with at least `ANTHROPIC_API_KEY` set (see
   **Required environment variables** below).

## Required Python version

⚠️ **Not verified in the repository** — there is no `pyproject.toml`, `runtime.txt`, or
`python_requires` pin anywhere in this project that states a minimum/required Python version.
The `.venv` currently checked into this working copy happens to have been created with
**Python 3.14.2**, but that reflects whatever the original developer had installed locally, not
a declared requirement. Based on the dependencies in `requirements.txt` (Streamlit ≥1.42,
pandas ≥2.2, etc.), a reasonably modern Python 3 (3.10+) is a safe practical assumption, but
this is a recommendation, not a verified constraint.

## How to install requirements

With your virtual environment activated:

```bash
pip install -r requirements.txt
```

Key dependencies include: `streamlit`, `pandas`, `numpy`, `plotly`, `python-dotenv`,
`reportlab` (PDF generation), `Faker` and `SQLAlchemy` (used by the separate data generator),
`strands-agents[anthropic]` / `strands-agents-tools` / `anthropic` (the agent framework), and
`boto3[crt]` (used for the SES email step). Optional Bedrock-related packages
(`bedrock-agentcore`, `bedrock-agentcore-starter-toolkit`) are also listed.

## Required environment variables

All configuration is read from a `.env` file in the project root. **Never commit real secret
values to this file** — only variable *names* are listed below.

| Variable | Required? | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** — no default, the app will error without it | Authenticates with the Claude/Anthropic API |
| `MES_MODEL_ID` | No (defaults to `claude-sonnet-4-6`) | Which Claude model all agents use |
| `MES_MAX_TOKENS` | No (defaults to `8296`, or `4096` via `startup.py`) | Max output tokens per model call |
| `MES_TEMPERATURE` | No (defaults to `0.2`) | Model sampling temperature |
| `MES_LOG_LEVEL` | No (defaults to `INFO`) | Logging verbosity (`DEBUG` shows rate-limit/retry detail) |
| `MES_LOG_FORMAT` | No (has a default format string) | Log line format |
| `MES_DB_PATH` | No (defaults to `mes.db` in the working directory) | Path to the SQLite MES database |
| `AWS_REGION` | No (defaults to `us-west-2`) | AWS region for the SES email client |
| `MES_SENDER_EMAIL` | No (has a placeholder default) | "From" address for notification email |
| `MES_RECIPIENT_EMAIL` | No (has a placeholder default) | "To" address for notification email |
| `MES_BASE_URL` | No (has a default demo URL) | Base URL used to build the PDF link in emails |
| `MES_API_TIMEOUT` | No (defaults to `40` seconds) | HTTP timeout for the Anthropic client |
| `MES_API_RETRIES` | No (defaults to `2`) | Max retries for the Anthropic client |
| `MES_EMAIL_DRY_RUN` | No (defaults to `true`) | When true, the notification step logs/returns a message instead of actually sending email |

⚠️ A few additional variables (`MES_MAX_RETRY_ATTEMPTS`, `MES_RETRY_MODE`, and commented-out AWS
credential variables) appear only in `mes_config.sh`, an alternate/legacy config script that
does not appear to be read by the current application code — **not verified as an active
requirement.**

## How to run the application using `python startup.py`

`startup.py` is a convenience bootstrap script that automates setup end-to-end:

```bash
python startup.py
```

It will:
1. Create a `.venv` virtual environment if one doesn't already exist.
2. Create a `.env` file if missing, prompting you interactively for your `ANTHROPIC_API_KEY`
   (and filling in default values for `MES_MODEL_ID`, `MES_MAX_TOKENS`, `MES_TEMPERATURE`).
3. Validate that an existing `.env` has a properly formatted API key, prompting again if not.
4. Install everything in `requirements.txt` into that virtual environment.
5. Launch the app with `streamlit run app.py`.

## How to run the Streamlit application (manually)

If you prefer to set things up yourself instead of using `startup.py`:

```bash
# from the project root, in Git Bash
source .venv/Scripts/activate        # Windows venv path is Scripts, not bin
python -m streamlit run app.py       # serves at http://localhost:8501
```

Ensure `.env` exists with at least `ANTHROPIC_API_KEY` set before starting.

## Location of the MES database

`mes.db` — a SQLite file at the project root (path configurable via `MES_DB_PATH`). It contains
14 tables (e.g. `Machines`, `WorkOrders`, `Downtimes`, `Defects`, `QualityControl`,
`OEEMetrics`) of synthetic data generated by a separate nested project
(`industrial-data-store-simulation-chatbot/app_factory/data_generator/`). This database is
intended to be treated as **read-only** by this application — the code path never issues
INSERT/UPDATE/DELETE against it — though that is enforced by coding convention, not by an
enforced read-only database connection mode.

## Location of generated reports

PDF reports are written to the `reports/` folder at the project root (created automatically if
it doesn't exist), with filenames following the pattern
`MES_Final_Report_YYYYMMDD_HHMMSS.pdf`.

## Current limitations

- **Synthetic data only.** All MES data is generated, not from a real factory — findings and
  reports are for demonstration purposes only.
- **Latency.** A full analysis run calls the Claude model multiple times in sequence (once per
  specialist agent plus the supervisor's own reasoning), so a single analysis can take several
  minutes to complete rather than returning instantly.
- **Report validation is still shallow.** The Verifier agent's confidence-scoring tool is a
  stub that returns a hardcoded confidence value rather than performing real
  schema/provenance-based validation. ⚠️ This means the "Verification Outcome" section of a
  generated report should be treated as a placeholder, not a rigorously validated confidence
  score — further work is needed here before trusting these reports for real decisions.
- **No real email is sent by default** (`MES_EMAIL_DRY_RUN=true`); the notification step only
  simulates sending.
- **Single-trigger design.** The UI intentionally allows only one way to start an analysis (the
  "Run Analysis" button), because the underlying agents are not safely re-entrant — this is a
  deliberate constraint, not a bug, but it does mean there's no way to queue or parallelize
  multiple analyses from the UI.
- **Computer-vision defect detection is not yet implemented.** The project roadmap describes
  replacing the manual defect-type dropdown with an image-based detection step (first a fake
  detection dictionary, later a real model), but this is not present in the current code — the
  dropdown is still the only selection method.
- ⚠️ *Not verified:* whether `startup.py` and `mes_config.sh` (which references AWS
  Bedrock-style configuration) are both actively maintained, or whether one is a legacy/
  experimental alternate path.

## Example workflow

1. Start the app (`python startup.py`, or the manual Streamlit command above) and open
   `http://localhost:8501`.
2. In the sidebar, set a look-back period (e.g. 7 days), leave "Maintenance Correlation"
   checked, and pick a defect type from the dropdown (e.g. "Battery Cell Variance").
3. Click **"🚀 Run Analysis"**.
4. The Supervisor agent calls Monitor → Analyzer → Planner → Verifier → Executor in sequence;
   progress/log output streams while this runs.
5. Once complete, the dashboard shows an Executive Summary and Performance Metrics for the run,
   and a PDF report appears in the `reports/` list (and is viewable via the app's PDF viewer).
6. The Executor agent's dry-run "email" confirms what would have been sent, without actually
   delivering anything.

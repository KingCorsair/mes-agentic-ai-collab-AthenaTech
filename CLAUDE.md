# CLAUDE.md

## What this project is

A multi-agent manufacturing defect analysis demo. A Streamlit UI lets a user pick a
defect type; a Supervisor agent (Strands SDK + Claude via the Anthropic API)
orchestrates five subagents — Monitor, Analyzer, Planner, Verifier, Executor — that
query a synthetic MES SQLite database (`mes.db`), identify likely root causes, and
produce a PDF report + (dry-run) email. All data is synthetic. This is a learning /
proof-of-concept project, not production software.

Planned next: a computer-vision defect detection front end will replace the manual
dropdown (fake detection dict first, then a real model), and database access will
move behind a read-only MCP server.

## How to run it

From Git Bash, in the project root:

```bash
source .venv/Scripts/activate        # Windows venv path: Scripts, not bin
python -m streamlit run app.py       # app at http://localhost:8501
```

All configuration is read from `.env` in the project root at import time.

## Hard rules — never violate these

1. **`mes.db` is READ-ONLY.** Never write to it, ALTER it, or "fix" its data.
   To get fresh data, rerun the generator in the nested
   `industrial-data-store-simulation-chatbot` repo
   (`app_factory/data_generator/sqlite-synthetic-mes-data.py`) and copy the new
   `mes.db` to the project root.
2. **Never hardcode config.** Model ID, max tokens, temperature, log level, and
   email settings all come from `.env` via `os.getenv` (see `MESAgentManager.__init__`).
   The model must stay switchable via `MES_MODEL_ID`.
3. **Every agent system prompt must end with the output-rules constant**
   (`OUTPUT_RULES` for subagents, `SUPERVISOR_OUTPUT_RULES` for the Supervisor),
   appended via `+` after the closing `"""`. New agents get it too. Never add
   prompt lines that contradict it (e.g. "provide confidence levels",
   "always quantify impact", "comprehensive") — that instructs fabrication.
4. **All SQL is parameterized.** User/agent input goes in as `?` placeholders,
   never string-formatted into the query. New predefined tools follow the house
   style: validate `days_back` (0–3650), compute `cutoff_date`, named columns
   (no `SELECT *`), `ORDER BY`, and a `LIMIT` cap.
5. **Do not add ways to start an analysis other than the Run button.**
   Auto-trigger paths in `app.py` caused duplicate concurrent agent invocations
   (Strands agents are NOT re-entrant). The button uses an `analysis_running`
   flag + a one-shot `work_pending` flag; keep that pattern.

## Database facts (verified with PRAGMA — trust these over intuition)

- 14 tables: BillOfMaterials, Defects, Downtimes, Employees, Inventory, Machines,
  MaterialConsumption, OEEMetrics, Products, QualityControl, Shifts, Suppliers,
  WorkCenters, WorkOrders.
- **`WorkOrders` has NO `ShiftID` column.** Shift is a property of the employee:
  join `WorkOrders → Employees (EmployeeID) → Shifts (e.ShiftID)`. Several repo
  queries were once broken by assuming a direct `wo.ShiftID`; don't reintroduce it.
- **There is NO Maintenance / maintenance_log / CMMS / quality_defects table.**
  Maintenance events are `Reason` values inside `Downtimes`
  ('Scheduled Maintenance', 'Cleaning', 'Software Error', 'Setup/Changeover'...).
- **`Defects` has NO date column.** Defect timing comes from
  `QualityControl.Date`, joined via `Defects.CheckID → QualityControl.CheckID`.
  `Defects` does have `RootCause` and `ActionTaken` columns — recorded ground truth.
- 36 distinct `DefectType` strings (e.g. 'Battery Cell Variance', 'Surface Defect').
  Defect types are not confined to the work center their name suggests.
- SQLite dialect only. No `TOP`, `DATEADD`, `NOW() - INTERVAL`, `DATE_SUB` — use
  `date(...)`, `julianday(...)`, parameterized cutoff dates.

## Architecture map

- `strands_agent.py` — everything agentic. `MESAgentManager`:
  - `_init_*_tools()` methods define per-agent `@tool` functions (docstrings are
    what the model sees — keep them precise about what table/granularity a tool returns).
  - `_init_agents()` builds the five subagents (prompts live here, ~line 1000+).
  - `_init_supervisor_agent()` wraps each subagent as a tool
    (`call_monitor_agent`, ...) — agents-as-tools, hub-and-spoke topology.
  - `_execute_safe_query()` runs parameterized SQL; the freeform `execute_sql`
    tool only permits an explicit allowlist (predefined tools bypass the
    allowlist because a human wrote and tested their SQL).
- `app.py` — Streamlit UI. Entire script reruns top-to-bottom on every user
  interaction; all long-running work must be guarded by the flag pattern above.
  All sidebar inputs are `disabled=` during a run.
- `reports/` — generated PDFs. `.env` — all config. `mes.db` — the database.

## Known gotchas (each of these cost real debugging time)

- **Streamlit reruns everything on any click/refresh.** Any unguarded
  "if condition: do work" will fire repeatedly. Guard with session-state flags;
  make work triggers one-shot.
- **Fan-out joins.** Any join between `Downtimes` and `WorkOrders` on
  `MachineID` alone multiplies rows ~20x (every downtime × every historical
  order on that machine). Always add a time-overlap condition:
  `AND dt.StartTime BETWEEN wo.ActualStartTime AND wo.ActualEndTime`.
- **`days_back` defaults must match the analysis scope** (UI default is 7).
  Tools defaulting to 30 quietly 4x the payload, burn the rate-limit budget,
  and cause long silent stalls (the client backs off invisibly at INFO level;
  set `MES_LOG_LEVEL=DEBUG` in `.env` to see 429s/retries).
- **Agents hallucinate table names** when they lack a sanctioned tool for a
  question (historically: `defect_log`, `maintenance_records`, `Maintenance`).
  Fix is tool coverage + the DATABASE FACTS lines in prompts, not looser SQL access.
- **LLM arithmetic is nondeterministic.** Totals/percentages must come from SQL
  (`SUM`, `COUNT`, `GROUP BY`), never from the model summing rows. If a number
  in a report can't be traced to a tool result, it's invented.
- **Windows/Git Bash:** files use CRLF line endings — preserve them; venv
  activation is `.venv/Scripts/activate`; paths contain spaces ("ATUL ANAND") —
  quote them in commands.
- `validate_findings` (Verifier tool) is a stub that returns a hardcoded
  0.85 confidence score — known repo debt, slated for replacement with real
  schema/provenance checks.

## How to verify a change didn't break things

Run one analysis (Battery Cell Variance, 7 days, Maintenance Correlation) and check:
1. Startup log shows the intended `Model ID` and `Max Tokens: 8296`.
2. Each `call_*_agent` appears exactly once in the log (no retry loops).
3. Zero `max_tokens stop reason` lines, zero `Query not in allowed list` warnings.
4. `analyze_downtime_correlations` returns well under 1,000 rows.
5. Full run completes in single-digit minutes.
6. Spot-check one number in the report against the DB, e.g.:
   `SELECT COUNT(*) FROM Defects d JOIN QualityControl qc ON d.CheckID=qc.CheckID
    WHERE d.DefectType='...' AND date(qc.Date) >= '...'`

## Roadmap context (so changes point the right direction)

1. Milestone 2: fake CV detection — hardcoded
   `{"label": ..., "confidence": ...}` dict + `LABEL_MAP` feeding the same
   `run_defect_analysis` path as the dropdown. CV output selects a defect type;
   it never writes to the database.
2. Milestone 3+: real `predict_defect(image_path)` from the CV repo replaces the dict.
3. PRD Phase 2: move DB access behind a read-only MCP server with fixed,
   parameterized tools (no open SQL tool).

<!-- TODO (owner): confirm startup.py loads .env / sets nothing that overrides it,
     and add the exact generator command + copy path you actually use. -->

"""
 Intelligent Agentic AI for Autonomous Manufacturing Operation 
Implements Monitor -> Analyzer -> Planner -> Verifier workflow with defect type selection
"""
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import json
import re
import logging
import queue
import threading
import time
from pathlib import Path
import urllib.parse

# # Add parent directory to path for imports
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

from strands_agent import MESAgentManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Langfuse-inspired shell: a constrained, centered content column; a bordered
# sidebar acting as the nav rail; soft slate borders on cards/expanders/metrics;
# and quiet, underlined tabs. Streamlit can't reproduce Langfuse's React shell
# exactly, but this approximates its clean, low-chrome look. Colors are the same
# shadcn "slate" values set in .streamlit/config.toml.
THEME_CSS = """
<style>
:root {
  --lf-border: #E2E8F0;      /* slate-200 */
  --lf-muted-fg: #64748B;    /* slate-500 */
  --lf-surface: #F8FAFC;     /* slate-50  */
}

/* Constrain and center the main content, like Langfuse's container pages. */
.block-container {
  max-width: 1120px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}

/* Sidebar as a bordered navigation rail. */
section[data-testid="stSidebar"] {
  background-color: var(--lf-surface);
  border-right: 1px solid var(--lf-border);
}

/* Cards, expanders and metrics get soft slate borders and a small radius. */
[data-testid="stExpander"],
[data-testid="stMetric"] {
  border: 1px solid var(--lf-border);
  border-radius: 10px;
  box-shadow: none;
}
[data-testid="stMetric"] {
  padding: 12px 16px;
  background: #FFFFFF;
}

/* Buttons: subtle border, gentle radius. */
.stButton > button,
.stDownloadButton > button {
  border-radius: 8px;
  border: 1px solid var(--lf-border);
}

/* Tabs: quiet, with an underlined active tab. */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  border-bottom: 1px solid var(--lf-border);
}
.stTabs [data-baseweb="tab"] {
  font-size: 0.92rem;
  padding-top: 8px;
  padding-bottom: 8px;
}

/* Breadcrumb line above the page title. */
.lf-breadcrumb {
  color: var(--lf-muted-fg);
  font-size: 0.8rem;
  letter-spacing: 0.01em;
  margin-bottom: 0.15rem;
}
</style>
"""


def inject_theme():
    """Apply the Langfuse-inspired shell styling (see THEME_CSS)."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

# Initialize session state
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = {}
if 'defect_types' not in st.session_state:
    st.session_state.defect_types = []
if 'selected_defect' not in st.session_state:
    st.session_state.selected_defect = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Create reports directory
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Human review decisions for reports, keyed by PDF filename. Lives next
# to the PDFs it describes (mes.db stays read-only).
DECISIONS_FILE = REPORTS_DIR / "report_decisions.json"


def load_report_decisions():
    """Recorded approve/reject decisions per report filename."""
    try:
        if DECISIONS_FILE.exists():
            return json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not read report decisions: {e}")
    return {}


def record_report_decision(pdf_filename: str, decision: str):
    """Persist a human review decision for a report."""
    decisions = load_report_decisions()
    decisions[pdf_filename] = {
        "decision": decision,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        DECISIONS_FILE.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Could not record report decision: {e}")

# JSON serialization helper
def datetime_to_string(obj):
    """Convert datetime objects to string for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def serialize_analysis_results(results):
    """Convert analysis results to JSON-serializable format"""
    if not isinstance(results, dict):
        return results
    
    serialized = {}
    for key, value in results.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, dict):
            serialized[key] = serialize_analysis_results(value)
        elif isinstance(value, list):
            serialized[key] = [serialize_analysis_results(item) if isinstance(item, dict) else datetime_to_string(item) for item in value]
        else:
            serialized[key] = value
    
    return serialized

# Initialize agent manager with caching

def _detect_app_base_url():
    """Best-effort public URL of this app from the current request, so
    emailed report links work for viewers other than the host machine."""
    try:
        headers = st.context.headers
        host = headers.get("Host")
        if not host:
            return None
        proto = headers.get("X-Forwarded-Proto")
        if not proto:
            proto = "http" if host.split(":")[0] in ("localhost", "127.0.0.1") else "https"
        return f"{proto}://{host}"
    except Exception:
        return None


def get_agent_manager():
    """Initialize and cache the MES Agent Manager"""
    try:
        # Explicit MES_BASE_URL config wins; otherwise point report links
        # at the address this app is actually being served from, so they
        # work on Streamlit Cloud without a manually configured secret.
        if not os.getenv("MES_BASE_URL"):
            detected = _detect_app_base_url()
            if detected:
                os.environ["MES_BASE_URL"] = detected
        manager = MESAgentManager()

        return manager
    except Exception as e:
        st.error(f"Failed to initialize agent manager: {e}")
        logger.error(f"Agent manager initialization error: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_defect_types(days_back: int = 365):
    """Load available defect types from the database"""
    try:
        agent_manager = get_agent_manager()
        if agent_manager is None:
            logger.error("Agent manager is None")
            return []
        
        logger.info(f"Loading defect types for last {days_back} days")
        result = agent_manager.get_defect_types(days_back)
        
        logger.info(f"Defect types result: {result}")
        
        # Check if we have rows (the function returns rows directly, not wrapped in success)
        if result and result.get('rows'):
            defect_types = [row['DefectType'] for row in result['rows'] if row.get('DefectType')]
            logger.info(f"Found {len(defect_types)} defect types: {defect_types}")
            return defect_types
        elif result and result.get('error'):
            logger.error(f"Database error: {result['error']}")
            st.error(f"Database error: {result['error']}")
            return []
        else:
            logger.warning("No defect types found in database")
            return []
            
    except Exception as e:
        logger.error(f"Error loading defect types: {e}")
        st.error(f"Error loading defect types: {e}")
        return []

def get_available_reports():
    """Get list of available PDF reports in the reports directory"""
    try:
        pdf_files = list(REPORTS_DIR.glob("*.pdf"))
        return sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)
    except Exception as e:
        logger.error(f"Error loading reports: {e}")
        return []

def get_pdf_from_url():
    """Check URL parameters for PDF file to display"""
    query_params = st.query_params
    pdf_file = query_params.get("pdf", None)
    
    if pdf_file:
        # Decode URL-encoded filename
        pdf_file = urllib.parse.unquote(pdf_file)
        pdf_path = REPORTS_DIR / pdf_file
        
        if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
            return pdf_path
    
    return None

def generate_pdf_url(pdf_filename):
    """Generate shareable URL for PDF file"""
    encoded_filename = urllib.parse.quote(pdf_filename)
    base_url = st.get_option("browser.serverAddress") or "localhost"
    port = st.get_option("server.port") or 8501
    
    # Get current URL components
    try:
        # Try to get the current URL from session state or use default
        current_url = f"http://{base_url}:{port}"
        pdf_url = f"{current_url}?pdf={encoded_filename}"
        return pdf_url
    except:
        # Fallback URL generation
        return f"?pdf={encoded_filename}"

def display_pdf_viewer(pdf_path):
    """Display a PDF inline using Streamlit's native PDF renderer.

    The old approach embedded the PDF as a data: base64 URI inside an
    <iframe>; current Chrome blocks rendering PDFs from data-URI iframes
    for security, showing 'not allowed to display'. st.pdf renders the
    bytes directly without that restriction. Returns the bytes so the
    caller can still offer a download button."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        try:
            st.pdf(pdf_data, height=800)
        except Exception as render_err:
            # Very old Streamlit without st.pdf, or a render hiccup: fall
            # back to a clear message rather than a blank/blocked frame.
            logger.warning(f"st.pdf failed, offering download only: {render_err}")
            st.info("Inline preview unavailable here — use the download button below to open the report.")

        return pdf_data
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        return None

# ---------------------------------------------------------------------------
# Observability event feed
#
# MESAgentManager emits structured event dicts (see strands_agent._emit) from
# SDK worker threads. The two renderers below draw the same events two ways:
# a compact live feed redrawn while the analysis thread works, and a full
# post-run "under the hood" trace for the pupil to explore.
# ---------------------------------------------------------------------------

AGENT_LABELS = {
    "monitor": "📡 Monitor Agent",
    "analyzer": "🔬 Analyzer Agent",
    "planner": "📋 Planner Agent",
    "verifier": "✅ Verifier Agent",
    "executor": "📧 Executor Agent",
    "supervisor": "🧠 Supervisor Agent",
}


def _agent_label(agent_name):
    return AGENT_LABELS.get(agent_name, f"🤖 {agent_name}")


def _plain_agent_name(agent_name):
    """Emoji-free name for use mid-sentence ('The Planner Agent wrote…')."""
    return f"{str(agent_name or 'supervisor').capitalize()} Agent"


def _format_structured(value, depth=0):
    """Markdown bullet lines for a dict/list tool input, so structured
    inputs read as a tidy indented list instead of raw JSON."""
    pad = "  " * depth
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}- **{key}**:")
                lines.extend(_format_structured(item, depth + 1))
            else:
                lines.append(f"{pad}- **{key}**: {item}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_format_structured(item, depth + 1))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}- {value}")
    return lines


def _group_events(events):
    """Fold the flat event list into a run header plus a chronological
    timeline of subagent activations and, between them, the Supervisor's
    own activity (its reasoning text and any direct tool use).

    Timeline entries are ("activation", dict) or ("supervisor", dict)."""
    run = {"started": None, "completed": None, "failed": None}
    timeline = []
    current = None

    for event in events:
        etype = event.get("type")
        payload = event.get("payload") or {}

        if etype == "run_started":
            run["started"] = event
        elif etype == "run_completed":
            run["completed"] = event
        elif etype == "run_failed":
            run["failed"] = event
        elif etype == "agent_started":
            current = {
                "agent_name": payload.get("agent_name"),
                "prompt": payload.get("delegation_prompt", ""),
                "items": [],
                "status": "running",
                "duration_s": None,
                "metrics": {},
                "result_preview": "",
                "error": None,
            }
            timeline.append(("activation", current))
        elif etype in ("agent_completed", "agent_failed"):
            if current is not None and payload.get("agent_name") == current["agent_name"]:
                if etype == "agent_completed":
                    current["status"] = "completed"
                    current["duration_s"] = payload.get("duration_s")
                    current["metrics"] = payload.get("metrics") or {}
                    current["result_preview"] = payload.get("result_preview", "")
                else:
                    current["status"] = "failed"
                    current["error"] = payload.get("error")
                current = None
        else:
            # The supervisor's call_*_agent tool events duplicate the
            # agent_started/agent_completed pair — skip them.
            tool_name = str(payload.get("tool_name") or "")
            if etype in ("tool_started", "tool_completed") and tool_name.startswith("call_"):
                continue
            if current is not None:
                current["items"].append(event)
            else:
                # Between delegations this is the Supervisor acting on its
                # own; group consecutive events into one timeline entry.
                if not (timeline and timeline[-1][0] == "supervisor"):
                    timeline.append(("supervisor", {"items": []}))
                timeline[-1][1]["items"].append(event)

    return run, timeline


def _render_sql_event(event):
    payload = event.get("payload") or {}
    purpose = payload.get("purpose")
    if purpose:
        st.caption(f"🗄️ The tool asked the MES database: “{purpose}” — as this SQL query:")
    if event.get("type") == "sql_failed":
        st.warning(f"🗄️ SQL failed: {payload.get('error')}")
        st.code(payload.get("sql", ""), language="sql", wrap_lines=True)
        return
    st.code(payload.get("sql", ""), language="sql", wrap_lines=True)
    params = payload.get("params")
    st.caption(
        f"🗄️ params: {params or '—'} → {payload.get('row_count')} rows"
        f" · {payload.get('execution_time_ms')} ms"
    )


def _render_event_items(items):
    """Render an activation's tool/SQL/guardrail events as feed lines.

    tool_started/tool_completed pairs (matched on tool_use_id) collapse
    into a single line. SQL events attach beneath the tool call that ran
    them (matched on the tool_use_id the backend stamps into them), so
    parallel tool calls can't interleave their queries; unstamped SQL
    from older event logs falls back to rendering in arrival order."""
    completed_by_id = {}
    sql_by_tool = {}
    for event in items:
        payload = event.get("payload") or {}
        if event.get("type") == "tool_completed":
            completed_by_id.setdefault(payload.get("tool_use_id"), event)
        elif event.get("type") in ("sql_executed", "sql_failed") and payload.get("tool_use_id"):
            sql_by_tool.setdefault(payload["tool_use_id"], []).append(event)

    for event in items:
        etype = event.get("type")
        payload = event.get("payload") or {}

        if etype == "tool_started":
            name = payload.get("tool_name")
            arguments = payload.get("arguments")
            # Split the tool's inputs three ways: short scalars stay inline
            # on the call line; long agent-written text renders below as
            # quoted prose; dict/list inputs render below as bullet lists.
            # Nothing is elided — every input appears in full somewhere,
            # so the call line never needs a truncation-looking ellipsis.
            long_texts = {}
            structured = {}
            if isinstance(arguments, dict):
                short = {}
                for key, value in arguments.items():
                    if isinstance(value, (dict, list)):
                        structured[key] = value
                    elif len(str(value)) > 200:
                        long_texts[key] = str(value)
                    else:
                        short[key] = value
                inline_args = ", ".join(f"{k}={v!r}" for k, v in short.items())
            else:
                # Events recorded before arguments were structured.
                text = " ".join(str(arguments or "").split())
                inline_args = text if len(text) <= 200 else "…"
                if inline_args != text:
                    long_texts = {"arguments": text}
            done = completed_by_id.get(payload.get("tool_use_id"))
            if done is None:
                st.markdown(f"🔧 `{name}({inline_args})` — running…")
            else:
                done_payload = done.get("payload") or {}
                duration = done_payload.get("duration_ms")
                duration_text = f" · {duration} ms" if duration is not None else ""
                if done_payload.get("status") == "error":
                    st.warning(f"🔧 `{name}({inline_args})` failed: {done_payload.get('error')}")
                else:
                    st.markdown(f"🔧 `{name}({inline_args})`{duration_text}")
            for key, text in long_texts.items():
                st.caption(
                    f"📝 The {_plain_agent_name(event.get('agent'))} wrote this "
                    f"text and passed it to the tool as `{key}`:"
                )
                st.markdown("> " + text.replace("\n", "\n> "))
            for key, value in structured.items():
                st.caption(
                    f"📋 The {_plain_agent_name(event.get('agent'))} filled in "
                    f"this structured input and passed it to the tool as `{key}`:"
                )
                st.markdown("\n".join("> " + line for line in _format_structured(value)))
            for sql_event in sql_by_tool.get(payload.get("tool_use_id"), []):
                _render_sql_event(sql_event)
        elif etype == "agent_message":
            text = payload.get("text", "")
            st.markdown("> " + text.replace("\n", "\n> "))
        elif etype in ("sql_executed", "sql_failed"):
            if not payload.get("tool_use_id"):
                _render_sql_event(event)
        elif etype == "guardrail_triggered":
            st.warning(
                f"🛡️ Guardrail **{payload.get('guardrail')}** blocked a query — "
                f"{payload.get('outcome')}"
            )
            st.code(payload.get("attempted_query", ""), language="sql", wrap_lines=True)
        elif etype == "agent_retry":
            st.warning(
                f"🔁 Attempt {payload.get('attempt')} failed, retrying: {payload.get('error')}"
            )


def _metrics_caption(metrics):
    if not metrics:
        return None
    parts = []
    if metrics.get("cycles") is not None:
        parts.append(f"{metrics['cycles']} model round-trips")
    if metrics.get("input_tokens") is not None:
        parts.append(f"{metrics['input_tokens']:,} tokens in / {metrics.get('output_tokens', 0):,} out")
    if metrics.get("model_seconds"):
        parts.append(f"{metrics['model_seconds']}s model time")
    return " · ".join(parts) if parts else None


def render_live_feed(events, running=True):
    """Redrawable live view of the run so far (call inside a fresh container).

    Two visual registers, kept strict so the class can tell them apart:
    small gray captions are the app narrating; quote blocks and code
    boxes are the actual text/SQL the agents and database exchanged."""
    run, timeline = _group_events(events)

    st.caption(
        "ℹ️ How to read this feed: gray notes like this one are the app "
        "explaining what is happening. Quoted blocks and code boxes are the "
        "actual messages the agents and the database sent each other."
    )

    if run["started"]:
        payload = run["started"]["payload"]
        window = ""
        if payload.get("window_start") and payload.get("window_end"):
            window = f" of data ({payload['window_start']} → {payload['window_end']})"
        st.caption(
            f"Run started — {payload.get('defect_type')}, last {payload.get('days_back')} days{window}"
            f" · scope: {payload.get('scope')}"
        )

    open_activation = False
    seen_activation = False
    for index, (kind, entry) in enumerate(timeline):
        if kind == "supervisor":
            if not seen_activation:
                label = "🧠 Supervisor Agent — planning the workflow"
            elif index == len(timeline) - 1 and (run["completed"] or run["failed"]):
                label = "🧠 Supervisor Agent — final synthesis"
            else:
                label = "🧠 Supervisor Agent — coordinating the next step"
            with st.status(label, state="complete", expanded=True):
                _render_event_items(entry["items"])
            continue

        seen_activation = True
        if entry["status"] == "running":
            open_activation = True
            label = f"{_agent_label(entry['agent_name'])} — working…"
            state = "running"
        elif entry["status"] == "failed":
            label = f"{_agent_label(entry['agent_name'])} — failed"
            state = "error"
        else:
            label = f"{_agent_label(entry['agent_name'])} — done in {entry['duration_s']}s"
            state = "complete"

        # Stay expanded after completion so the class can review each
        # agent's steps without re-opening every box.
        with st.status(label, state=state, expanded=True):
            prompt = str(entry["prompt"] or "").strip()
            if prompt:
                st.caption(
                    f"📨 The Supervisor wrote this task briefing and handed it "
                    f"to the {_plain_agent_name(entry['agent_name'])}:"
                )
                st.markdown("> " + prompt.replace("\n", "\n> "))
            _render_event_items(entry["items"])
            if entry["error"]:
                st.error(entry["error"])
            footer = _metrics_caption(entry["metrics"])
            if footer:
                st.caption(footer)

    if run["completed"]:
        payload = run["completed"]["payload"]
        st.success(
            f"Analysis complete in {payload.get('duration_s')}s — "
            f"report {payload.get('pdf_filename')}"
        )
    elif run["failed"]:
        st.error(f"Analysis failed: {run['failed']['payload'].get('error')}")
    elif running and not open_activation:
        # Between delegations the Supervisor itself is reasoning.
        label = ("🧠 Supervisor Agent — synthesizing…" if any(k == "activation" for k, _ in timeline)
                 else "🧠 Supervisor Agent — planning the workflow…")
        st.status(label, state="running")


def render_trace(events):
    """Post-run replay of the run in exactly the live feed's format, so
    the reader never has to learn a second layout. The PDF report and the
    run folder's events.jsonl hold the full unabridged record."""
    st.caption(
        "Every step the agent system took, in order — the same view you "
        "watched during the run. Prompts flow down from the Supervisor to "
        "the subagents; each subagent's tools run SQL against the MES "
        "database, and findings flow back up."
    )

    run, _ = _group_events(events)
    if run["started"]:
        payload = run["started"]["payload"]
        st.markdown(f"**Run id:** `{payload.get('run_id')}`")

    render_live_feed(events, running=False)

    if run["completed"]:
        payload = run["completed"]["payload"]
        st.markdown(
            f"**Outcome:** completed in {payload.get('duration_s')}s — "
            f"final report `{payload.get('pdf_filename')}`"
        )
    elif run["failed"]:
        st.error(f"Run failed: {run['failed']['payload'].get('error')}")


def run_defect_analysis(defect_type: str, days_back: int = 7, include_oee: bool = True,
                        include_downtime: bool = True, include_changeover: bool = True,
                        include_maintenance: bool = True, render_fn=None):
    """Execute the supervisor workflow in a background thread and stream events
    to a live renderer, then validate and store the result.

    Backend behavior is unchanged: same agent_manager.run_defect_analysis call,
    same thread-safe event queue, same PDF validation. Only the presentation is
    parameterized — `render_fn(events, running)` draws the live view (defaults to
    the detailed feed; the redesigned UI passes a humanized st.status renderer).

    Errors are recorded in st.session_state['run_error'] (rendered as plain
    English by the caller with Retry/Reset) rather than drawn here, and the full
    event log is always preserved in session state. Returns the result dict or
    None on failure.
    """
    agent_manager = get_agent_manager()
    if agent_manager is None:
        st.session_state.run_error = (
            "The investigation service is unavailable. Please refresh the page and try again."
        )
        return None

    render_fn = render_fn or render_live_feed

    try:
        # The manager emits observability events from SDK worker threads, which
        # must never touch Streamlit directly. Events go into a thread-safe
        # queue; the analysis runs in a background thread; this (main) thread
        # polls the queue and redraws the live view.
        event_queue = queue.Queue()
        agent_manager.on_event = event_queue.put

        run_outcome = {}

        def _analysis_worker():
            try:
                run_outcome["result"] = agent_manager.run_defect_analysis(
                    defect_type=defect_type,
                    days_back=days_back,
                    include_oee=include_oee,
                    include_downtime=include_downtime,
                    include_changeover=include_changeover,
                    include_maintenance=include_maintenance
                )
            except Exception as worker_error:
                run_outcome["error"] = worker_error

        worker = threading.Thread(target=_analysis_worker, daemon=True)
        worker.start()

        feed_placeholder = st.empty()
        events = []
        with feed_placeholder.container():
            render_fn(events, True)

        while worker.is_alive() or not event_queue.empty():
            drained = False
            while True:
                try:
                    events.append(event_queue.get_nowait())
                    drained = True
                except queue.Empty:
                    break
            if drained:
                with feed_placeholder.container():
                    render_fn(events, True)
            time.sleep(0.4)

        worker.join()
        agent_manager.on_event = None

        # Always preserve the full event log for the technical tabs and history,
        # even on failure (so a failed run's steps stay visible).
        st.session_state.agent_event_log = events
        with feed_placeholder.container():
            render_fn(events, False)

        if "error" in run_outcome:
            st.session_state.run_error = str(run_outcome["error"])
            return None

        analysis_results = run_outcome.get("result")
        if not isinstance(analysis_results, dict):
            st.session_state.run_error = "No result was returned by the investigation."
            return None

        if analysis_results.get("status") != "completed":
            st.session_state.run_error = analysis_results.get(
                "error", "The investigation did not complete successfully."
            )
            return None

        pdf_filename = analysis_results.get("pdf_filename")
        pdf_path_value = analysis_results.get("pdf_path")
        if not pdf_filename or not pdf_path_value:
            st.session_state.run_error = "The investigation completed but produced no report file."
            logger.error("Completed analysis missing pdf info: %s", analysis_results)
            return None

        pdf_path = Path(pdf_path_value)
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            st.session_state.run_error = (
                "The investigation completed but the report file is missing or empty."
            )
            logger.error("Generated PDF missing/empty: %s", pdf_path)
            return None

        # Store results only after verifying the PDF.
        st.session_state.current_analysis = analysis_results
        st.session_state.analysis_started = True
        st.session_state.pop("run_error", None)
        logger.info("Analysis completed with report: %s", pdf_path)
        return analysis_results

    except Exception as e:
        st.session_state.run_error = f"Error during investigation: {e}"
        logger.error(f"Analysis error: {e}")
        return None
    finally:
        if agent_manager is not None:
            agent_manager.on_event = None

def check_analysis_window(defect_type: str, days_back: int):
    """Pre-run check: refuse to launch the agents into a window that
    provably holds no records for this defect. Returns (ok, warning_text).
    Fails open — a broken pre-check must never block a run."""
    try:
        agent_manager = get_agent_manager()
        if agent_manager is None:
            return True, ""
        result = agent_manager.get_defect_window_stats(defect_type, days_back)
        if not result.get("success") or not result.get("rows"):
            return True, ""
        row = result["rows"][0]
        if (row.get("WindowCount") or 0) > 0:
            return True, ""
        last = row.get("LastOccurrence")
        anchor = getattr(agent_manager, "data_anchor_date", None)
        if last and anchor is not None:
            last_date = pd.to_datetime(last)
            gap_days = (pd.Timestamp(anchor) - last_date).days
            return False, (
                f"⏳ No **{defect_type}** records exist in the last {days_back} days "
                f"of data — the most recent is from **{last_date.strftime('%Y-%m-%d')}**. "
                f"Widen the look-back period to at least {gap_days + 1} days, "
                f"then press Run Analysis again."
            )
        return False, (
            f"⏳ No **{defect_type}** records exist anywhere in the database, "
            f"so an analysis would have nothing to work with."
        )
    except Exception as e:
        logger.warning(f"Analysis window pre-check failed: {e}")
        return True, ""


def render_defect_selection():
    """Render defect type selection interface in sidebar"""
    
    st.sidebar.subheader("🎯 Start here — Select a defect")
    st.sidebar.info("Pick a defect type to analyze. Then press **Run Analysis** to start the AI workflow for it.")
    
    # Load defect types
    with st.sidebar:
        with st.spinner("Loading available defect types..."):
            defect_types = load_defect_types(365)  # Load from last 365 days
            st.session_state.defect_types = defect_types
        


        
        if not defect_types:
            st.warning("⚠️ No defect types found in the database. Please check your data connection.")
            
            # Show database connection status
            agent_manager = get_agent_manager()
            if agent_manager:
                test_result = agent_manager.test_database_connection()
                if test_result.get('success'):
                    st.info(f"✅ Database connected: {test_result.get('total_tables', 0)} tables found")
                    for table, count in test_result.get('tables', {}).items():
                        st.write(f"- {table}: {count} records")
                else:
                    st.error(f"❌ Database connection failed: {test_result.get('error')}")
            
            return None
        
        # Defect type dropdown. The widget's own state is destroyed whenever
        # the sidebar doesn't render (e.g. the PDF viewer's early return), so
        # restore the previous choice from session state instead of index=0.
        options = [None] + defect_types
        previous = st.session_state.get("selected_defect")
        selected_defect = st.selectbox(
            "Select Event Type for Analysis:",
            options=options,
            index=options.index(previous) if previous in options else 0,
            disabled=st.session_state.analysis_running,
            format_func=lambda x: "-- Select an Event type --" if x is None else x,
            help="Choose a specific defect type to analyze using the AI agent workflow"
        )
        
        st.session_state.selected_defect = selected_defect
        
    return selected_defect

@st.cache_data(ttl=300)
def get_defect_overview(days_back: int = 30):
    """Real, read-only overview data for the landing dashboard.

    Three parameterized SELECTs against the read-only MES database, anchored to
    the newest record (not today), following the repo's query house style. All
    figures shown on the overview trace back to these queries — nothing is
    estimated. Returns a dict of DataFrames plus headline totals, or None.
    """
    agent_manager = get_agent_manager()
    if agent_manager is None:
        return None
    try:
        anchor = agent_manager.data_anchor_date
        anchor_str = anchor.strftime('%Y-%m-%d')
        cutoff_str = agent_manager._cutoff_date(days_back)
        window = (cutoff_str, anchor_str)

        by_type = agent_manager._execute_safe_query(
            """SELECT d.DefectType AS DefectType, COUNT(*) AS Occurrences
               FROM Defects d JOIN QualityControl qc ON d.CheckID = qc.CheckID
               WHERE date(qc.Date) >= ? AND date(qc.Date) <= ?
               GROUP BY d.DefectType ORDER BY Occurrences DESC LIMIT 10""",
            window, purpose="Top defect types in the overview window")

        by_severity = agent_manager._execute_safe_query(
            """SELECT d.Severity AS Severity, COUNT(*) AS Occurrences
               FROM Defects d JOIN QualityControl qc ON d.CheckID = qc.CheckID
               WHERE date(qc.Date) >= ? AND date(qc.Date) <= ?
               GROUP BY d.Severity ORDER BY d.Severity""",
            window, purpose="Defect severity distribution in the overview window")

        by_day = agent_manager._execute_safe_query(
            """SELECT date(qc.Date) AS Day, COUNT(*) AS Occurrences
               FROM Defects d JOIN QualityControl qc ON d.CheckID = qc.CheckID
               WHERE date(qc.Date) >= ? AND date(qc.Date) <= ?
               GROUP BY Day ORDER BY Day""",
            window, purpose="Defects per day in the overview window")

        if not (by_type.get("success") and by_day.get("success")):
            return None

        type_df = by_type["dataframe"]
        sev_df = by_severity["dataframe"] if by_severity.get("success") else pd.DataFrame()
        day_df = by_day["dataframe"]

        total = int(type_df["Occurrences"].sum()) if not type_df.empty else 0
        # Weighted mean severity across the distribution.
        avg_sev = 0.0
        if not sev_df.empty and sev_df["Occurrences"].sum() > 0:
            avg_sev = float((sev_df["Severity"] * sev_df["Occurrences"]).sum() / sev_df["Occurrences"].sum())

        return {
            "anchor": anchor_str,
            "cutoff": cutoff_str,
            "days_back": days_back,
            "by_type": type_df,
            "by_severity": sev_df,
            "by_day": day_df,
            "total_defects": total,
            "distinct_types": int(type_df.shape[0]),
            "avg_severity": avg_sev,
            "top_type": type_df.iloc[0]["DefectType"] if not type_df.empty else "—",
        }
    except Exception as e:
        logger.warning(f"Defect overview query failed: {e}")
        return None


def render_overview_dashboard():
    """Charts-first operational overview, shown at the top of the landing page.

    Mirrors the useful shape of an ops dashboard: headline KPIs, then defect
    mix and severity side by side, then a trend line — all from live MES data.
    """
    overview = get_defect_overview(30)
    if not overview or overview["total_defects"] == 0:
        return

    st.subheader("📊 Quality Overview")
    st.caption(
        f"Last {overview['days_back']} days of data "
        f"({overview['cutoff']} → {overview['anchor']}, ending at the newest "
        f"record in the database)."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Defects Recorded", f"{overview['total_defects']:,}")
    k2.metric("Defect Types", f"{overview['distinct_types']}")
    sev = overview["avg_severity"]
    sev_word = "High" if sev > 3 else "Medium" if sev > 2 else "Low"
    k3.metric("Avg Severity", f"{sev:.1f}/5 ({sev_word})")
    k4.metric("Most Frequent", overview["top_type"])

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Top defect types**")
        type_df = overview["by_type"].sort_values("Occurrences")
        fig = px.bar(
            type_df, x="Occurrences", y="DefectType", orientation="h",
            color_discrete_sequence=["#0B6BCB"],
        )
        fig.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("**Severity distribution**")
        sev_df = overview["by_severity"].copy()
        if not sev_df.empty:
            sev_df["Label"] = sev_df["Severity"].map(lambda s: f"Sev {int(s)}")
            fig = px.bar(
                sev_df, x="Label", y="Occurrences",
                color="Severity", color_continuous_scale="Blues",
            )
            fig.update_layout(
                template="plotly_white", height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title=None, yaxis_title=None, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No severity data in this window.")

    st.markdown("**Defects per day**")
    day_df = overview["by_day"]
    fig = px.area(day_df, x="Day", y="Occurrences", color_discrete_sequence=["#0B6BCB"])
    fig.update_traces(line_color="#0B6BCB", fillcolor="rgba(11,107,203,0.12)")
    fig.update_layout(
        template="plotly_white", height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None, yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.divider()


def render_defect_preview(defect_type: str):
    """Render detailed preview information for selected defect type in main area"""
    
    try:
        agent_manager = get_agent_manager()
        if agent_manager is None:
            return
        
        result = agent_manager.get_defect_preview(defect_type)
        
        if result and result.get('rows') and len(result['rows']) > 0:
            data = result['rows'][0]

            # The preview query counts a fixed 30-day window; the analysis
            # uses the sidebar's look-back. Label the window so the two
            # never look contradictory (e.g. 412 here vs 0 in a 7-day run).
            anchor = getattr(agent_manager, "data_anchor_date", None)
            if anchor is not None:
                st.caption(
                    f"All numbers below cover the 30 days of data ending "
                    f"{anchor:%Y-%m-%d} (the newest record in the database)."
                )
            else:
                st.caption("All numbers below cover the last 30 days of records.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Occurrences", f"{data.get('TotalOccurrences', 0):,}")
                st.metric("Machines Affected", f"{data.get('MachinesAffected', 0):,}")

            with col2:
                severity = data.get('AvgSeverity') or 0
                severity_word = "High" if severity > 3 else "Medium" if severity > 2 else "Low"
                severity_color = "🔴" if severity > 3 else "🟡" if severity > 2 else "🟢"
                st.metric("Avg Severity", f"{severity_color} {severity:.1f}/5 ({severity_word})")
                st.metric("Products Affected", f"{data.get('ProductsAffected', 0):,}")
            
            with col3:
                st.metric("Root Cause Variety", f"{data.get('RootCauseVariety', 0):,}")
                last_occurrence = data.get('LastOccurrence')
                if last_occurrence:
                    try:
                        last_date = pd.to_datetime(last_occurrence).strftime('%Y-%m-%d')
                        st.metric("Last Occurrence", last_date)
                    except:
                        st.metric("Last Occurrence", str(last_occurrence))
                else:
                    st.metric("Last Occurrence", "Unknown")
            
            # Risk assessment
            risk_score = 0
            if severity > 3:
                risk_score += 2
            if data.get('TotalOccurrences', 0) > 10:
                risk_score += 1
            if data.get('MachinesAffected', 0) > 3:
                risk_score += 1
            
            if risk_score >= 3:
                risk_level = "🔴 High"
            elif risk_score >= 2:
                risk_level = "🟡 Medium"
            else:
                risk_level = "🟢 Low"
            
            st.info(f"**Risk Assessment:** {risk_level} | **Recommendation:** {'Immediate analysis recommended' if risk_score >= 3 else 'Standard analysis sufficient'}")
            
        else:
            st.warning(f"No preview data available for defect type: {defect_type}")
            if result and result.get('error'):
                st.error(f"Database error: {result['error']}")
            
    except Exception as e:
        st.error(f"Error loading defect preview: {e}")
        logger.error(f"Defect preview error: {e}")


# ===========================================================================
# Redesigned UI: a guided investigation workspace.
#
# Navigation (native st.navigation): Investigate / Run History / How It Works.
# Sidebar holds settings only. Case selection lives on the Investigate page.
# Backend behavior is untouched — every function below reuses the existing
# helpers (run_defect_analysis, get_defect_preview, _group_events, the report
# decision log, PDF viewer, etc.). Raw JSON and SQL are hidden by default.
# ===========================================================================

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Sample investigations for the empty state (real defect types are loaded live;
# these are only shown as illustrative starting points).
SAMPLE_INVESTIGATIONS = [
    "Waterproofing Failure",
    "Control Board Error",
    "Sensor Malfunction",
]


def init_session_state():
    """Centralized, rerun-safe defaults for every key the UI relies on.

    Uses setdefault so a completed run is never wiped by a later rerun, and so
    widget-bound keys (toggles, radios) exist before their widgets render.
    """
    defaults = {
        # existing keys (preserved)
        "analysis_started": False,
        "current_analysis": {},
        "defect_types": [],
        "selected_defect": None,
        "analysis_running": False,
        "work_pending": False,
        "agent_event_log": [],
        "show_final_analysis": False,
        # new keys
        "investigation_mode": "record",
        "user_question": "",
        "run_params": {},
        "approval_state": None,
        "selected_history_row": None,
        "learning_mode": False,
        "show_technical_data": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Derived facts from the event log (all real; nothing invented)
# ---------------------------------------------------------------------------

def _tool_calls(events):
    """Count real tool invocations, excluding the supervisor's call_*_agent
    delegations (those duplicate the agent activations)."""
    return sum(
        1 for e in events
        if e.get("type") == "tool_started"
        and not str((e.get("payload") or {}).get("tool_name", "")).startswith("call_")
    )


def _records_examined(events):
    """Total rows returned across all successful SQL queries."""
    return sum(
        int((e.get("payload") or {}).get("row_count") or 0)
        for e in events if e.get("type") == "sql_executed"
    )


def _confidence_from_events(events):
    """Best-effort verifier confidence, only if the verifier actually reported
    one. Returns None otherwise (shown as 'Not scored') — never fabricated.

    Note: the current verifier tool is a stub returning a fixed score, so this
    is surfaced honestly and may read as a constant; see the backend notes.
    """
    for e in events:
        payload = e.get("payload") or {}
        blob = f"{payload}"
        if "confidence_score" in blob:
            m = re.search(r"confidence_score['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)", blob)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


def _extract_report_sections(markdown_text):
    """Re-slice the Supervisor's own markdown report into labelled sections by
    heading. This only reorganizes text the agents produced — it never invents
    content. Falls back to a single 'conclusion' blob if no headings match.
    """
    if not markdown_text:
        return {}
    heading_map = [
        (r"root\s*cause", "root_cause"),
        (r"recommend|action|next step", "recommendation"),
        (r"evidence|finding|analysis", "evidence"),
        (r"verif|limitation|concern|caveat|uncertain", "limitations"),
    ]
    sections = {}
    current = "conclusion"
    for line in markdown_text.splitlines():
        stripped = line.strip()
        heading = re.match(r"^#{1,6}\s*(.+?)\s*#*$", stripped) or re.match(r"^\*\*(.+?)\*\*:?$", stripped)
        if heading:
            title = heading.group(1).lower()
            current = "other"
            for pattern, key in heading_map:
                if re.search(pattern, title):
                    current = key
                    break
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


# ---------------------------------------------------------------------------
# Section 2 — humanized live workflow (st.status)
# ---------------------------------------------------------------------------

def _humanize_timeline(events):
    """Fold raw events into concise, user-facing workflow steps."""
    run, timeline = _group_events(events)
    steps = []

    if run["started"]:
        p = run["started"]["payload"]
        steps.append({
            "state": "complete", "agent": "Supervisor",
            "title": "Supervisor prepared the investigation plan",
            "detail": f"{p.get('defect_type')} · last {p.get('days_back')} days "
                      f"({p.get('window_start')} → {p.get('window_end')})",
            "duration": None, "raw": run["started"],
        })

    for kind, entry in timeline:
        if kind == "supervisor":
            continue  # supervisor coordination is shown in the trace tab, not the concise feed
        agent = _plain_agent_name(entry.get("agent_name"))
        rows = _records_examined(entry.get("items", []))
        tools = _tool_calls(entry.get("items", []))
        status = entry.get("status")
        state = {"completed": "complete", "failed": "failed", "running": "running"}.get(status, "running")
        if state == "running":
            title = f"{agent} is working…"
        elif rows:
            title = f"{agent} reviewed {rows:,} records"
        elif tools:
            title = f"{agent} ran {tools} check{'s' if tools != 1 else ''}"
        else:
            title = f"{agent} completed its step"
        steps.append({
            "state": state, "agent": agent, "title": title,
            "detail": (entry.get("result_preview") or "").strip()[:180],
            "duration": entry.get("duration_s"),
            "error": entry.get("error"), "raw": entry,
        })

    if run["completed"]:
        p = run["completed"]["payload"]
        steps.append({
            "state": "complete", "agent": "Supervisor",
            "title": "Investigation complete — report generated",
            "detail": p.get("pdf_filename", ""), "duration": p.get("duration_s"),
            "raw": run["completed"],
        })
    elif run["failed"]:
        steps.append({
            "state": "failed", "agent": "Supervisor",
            "title": "Investigation failed",
            "detail": run["failed"]["payload"].get("error", ""),
            "duration": None, "raw": run["failed"],
        })
    return run, steps


_STATE_ICON = {
    "complete": "✅", "running": "⏳", "failed": "❌",
    "warning": "⚠️", "pending": "◻️", "approval": "🟡",
}


def render_live_workflow(events, running=True):
    """Concise, human-readable workflow view built on st.status.

    Used both as the live renderer during a run and as the collapsed post-run
    summary. Raw events appear only when 'Show technical data' is enabled.
    """
    run, steps = _humanize_timeline(events)

    if run.get("failed"):
        label, state = "Investigation failed", "error"
    elif run.get("completed"):
        label, state = "Investigation complete", "complete"
    elif running:
        label, state = "Agents are investigating…", "running"
    else:
        label, state = "Investigation finished", "complete"

    with st.status(label, state=state, expanded=bool(running)):
        if not steps:
            st.write("Preparing the investigation…")
        for step in steps:
            icon = _STATE_ICON.get(step["state"], "•")
            duration = f" · {step['duration']}s" if step.get("duration") else ""
            st.markdown(
                f"{icon} **{step['title']}**  \n"
                f"<span style='color:#6b7280;font-size:0.85em'>{step['agent']}{duration}</span>",
                unsafe_allow_html=True,
            )
            if step.get("detail"):
                st.caption(step["detail"])
            if step.get("error"):
                st.error(step["error"])
            if st.session_state.get("show_technical_data"):
                with st.expander("Raw event"):
                    st.json(step.get("raw"))
        if running and not run.get("completed") and not run.get("failed"):
            st.caption("Working… this typically takes a few minutes.")


# ---------------------------------------------------------------------------
# Sidebar (settings only)
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🏭 MES Investigator")
        st.caption("AthenaTech educational demo")
        st.divider()

        st.subheader("Settings")
        st.toggle("Learning mode", key="learning_mode",
                  help="Show short 'why this matters' explanations.")
        st.toggle("Show technical data", key="show_technical_data",
                  help="Reveal raw events, SQL and JSON throughout the app.")

        model = os.getenv("MES_MODEL_ID")
        if model:
            st.caption(f"Model: `{model}`")

        st.divider()
        st.subheader("Current run")
        if st.session_state.get("analysis_running"):
            st.info("⏳ Investigation running…")
        elif st.session_state.get("run_error"):
            st.error("❌ Last run failed")
        elif st.session_state.get("analysis_started"):
            st.success("✅ Investigation complete")
        else:
            st.caption("No active investigation")

        if st.button("Reset current investigation", use_container_width=True,
                     key="reset_investigation"):
            for key in ["analysis_started", "selected_defect", "defect_select",
                        "show_final_analysis", "agent_event_log", "work_pending",
                        "analysis_running", "run_error", "approval_state",
                        "user_question", "run_params"]:
                st.session_state.pop(key, None)
            st.session_state.current_analysis = {}
            st.rerun()


# ---------------------------------------------------------------------------
# Section 1 — choose an investigation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_defect_example_record(defect_type: str):
    """One real, most-recent record of this defect type, for the case preview.

    The workflow analyzes a defect *type*, so this shows a representative record
    (id, machine, line, severity, detection time) drawn live from the database.
    Read-only, parameterized, fail-open.
    """
    manager = get_agent_manager()
    if manager is None or not defect_type:
        return None
    try:
        result = manager._execute_safe_query(
            """SELECT d.DefectID AS DefectID, d.Severity AS Severity,
                      qc.Date AS DetectedAt, m.Name AS Machine, wc.Name AS Line
               FROM Defects d
               JOIN QualityControl qc ON d.CheckID = qc.CheckID
               LEFT JOIN WorkOrders wo ON qc.OrderID = wo.OrderID
               LEFT JOIN Machines m ON wo.MachineID = m.MachineID
               LEFT JOIN WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
               WHERE d.DefectType = ?
               ORDER BY date(qc.Date) DESC LIMIT 1""",
            (defect_type,), purpose="Most recent example record for the case preview")
        if result.get("success") and result.get("rows"):
            return result["rows"][0]
    except Exception as e:
        logger.warning(f"Example record lookup failed: {e}")
    return None


def render_selected_case_preview(defect_type: str):
    """Compact preview of the selected case before execution."""
    record = get_defect_example_record(defect_type)
    with st.container(border=True):
        st.markdown(f"**Selected case — {defect_type}**")
        if record:
            detected = str(record.get("DetectedAt") or "")[:16]
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Defect ID**\n\n{record.get('DefectID', '—')}")
            c1.markdown(f"**Defect type**\n\n{defect_type}")
            c2.markdown(f"**Machine**\n\n{record.get('Machine') or '—'}")
            c2.markdown(f"**Production line**\n\n{record.get('Line') or '—'}")
            c3.markdown(f"**Severity**\n\n{record.get('Severity', '—')}/5")
            c3.markdown(f"**Detected**\n\n{detected or '—'}")
            st.caption("Shown: most recent record of this type. The investigation "
                       "analyzes the defect type across the selected look-back window.")
        else:
            st.caption("No example record available for this defect type yet.")


def render_investigation_form():
    """Section 1: choose a defect/record or ask a question, then run."""
    with st.container(border=True):
        st.markdown("### 1 · Choose an investigation")

        # Mode selector and live inputs live outside the form so the case
        # preview can update immediately; the form submits scope + run together.
        st.radio(
            "How would you like to start?",
            options=["record", "question"],
            format_func=lambda m: "Select a defect record" if m == "record"
            else "Ask an investigation question",
            horizontal=True, key="investigation_mode",
        )

        defect_types = st.session_state.get("defect_types") or []
        if not defect_types:
            with st.spinner("Loading defect types…"):
                defect_types = load_defect_types(365)
                st.session_state.defect_types = defect_types

        if not defect_types:
            st.warning("No defect types are available. Check the database connection.")
            return

        options = [None] + defect_types
        previous = st.session_state.get("selected_defect")

        if st.session_state.investigation_mode == "question":
            st.text_area(
                "Your investigation question",
                placeholder="e.g. Why did waterproofing failures rise last week?",
                key="user_question",
            )
            st.caption("The engine investigates a specific defect type — pick the "
                       "one your question is about.")

        chosen = st.selectbox(
            "Defect type" if st.session_state.investigation_mode == "record"
            else "Which defect does this concern?",
            options=options,
            index=options.index(previous) if previous in options else 0,
            format_func=lambda x: "— Select a defect type —" if x is None else x,
            disabled=st.session_state.analysis_running,
            key="defect_select",
        )
        st.session_state.selected_defect = chosen

        if chosen:
            render_selected_case_preview(chosen)

        with st.form("investigation_form"):
            st.markdown("**Look-back window & scope**")
            col1, col2 = st.columns([1, 3])
            with col1:
                time_option = st.selectbox(
                    "Look-back period",
                    ["Last 3 days", "Last 7 days", "Last 14 days", "Last 30 days",
                     "Last 120 days", "Last 180 days", "Last 365 days"],
                    index=1, key="lookback_select",
                )
            with col2:
                sc1, sc2, sc3, sc4 = st.columns(4)
                include_oee = sc1.checkbox("OEE", value=False, key="scope_oee")
                include_downtime = sc2.checkbox("Downtime", value=False, key="scope_downtime")
                include_changeover = sc3.checkbox("Changeover", value=False, key="scope_changeover")
                include_maintenance = sc4.checkbox("Maintenance", value=True, key="scope_maintenance")

            submitted = st.form_submit_button(
                "Run Investigation →", type="primary", use_container_width=True,
                disabled=st.session_state.analysis_running,
            )

        if submitted:
            if not st.session_state.selected_defect:
                st.warning("Please choose a defect type to investigate.")
                return
            days_back = int(time_option.split()[1])
            window_ok, window_warning = check_analysis_window(
                st.session_state.selected_defect, days_back)
            if not window_ok:
                st.warning(window_warning)
                return
            st.session_state.run_params = {
                "days_back": days_back,
                "include_oee": include_oee,
                "include_downtime": include_downtime,
                "include_changeover": include_changeover,
                "include_maintenance": include_maintenance,
            }
            st.session_state.analysis_running = True
            st.session_state.work_pending = True
            st.session_state.analysis_started = False
            st.session_state.current_analysis = {}
            st.session_state.approval_state = None
            st.session_state.pop("run_error", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Section 3 — investigation result (conclusion before technical detail)
# ---------------------------------------------------------------------------

def render_result_summary(analysis):
    """The plain-language conclusion and headline metrics."""
    markdown_text = analysis.get("supervisor_orchestration") or ""
    sections = _extract_report_sections(markdown_text)
    events = st.session_state.get("agent_event_log") or []

    st.markdown("#### Most likely root cause")
    with st.container(border=True):
        st.markdown(sections.get("root_cause") or sections.get("conclusion")
                    or markdown_text or "_No conclusion text was returned._")

    recommendation = sections.get("recommendation")
    if recommendation:
        st.markdown("#### Recommended action")
        with st.container(border=True):
            st.markdown(recommendation)

    limitations = sections.get("limitations")
    if limitations:
        st.markdown("#### Verifier concerns & limitations")
        with st.container(border=True):
            st.markdown(limitations)

    # Four bordered metric tiles (short values only).
    confidence = _confidence_from_events(events)
    tiles = [
        ("Confidence", f"{confidence * 100:.0f}%" if confidence is not None else "Not scored"),
        ("Records examined", f"{_records_examined(events):,}"),
        ("Tool calls", f"{_tool_calls(events)}"),
        ("Total runtime", f"{analysis.get('total_duration', 0):.1f}s"),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, tiles):
        with col:
            with st.container(border=True):
                st.metric(label, value)


# ---------------------------------------------------------------------------
# Human approval (st.dialog)
# ---------------------------------------------------------------------------

@st.dialog("Human approval required")
def _approval_dialog(analysis):
    sections = _extract_report_sections(analysis.get("supervisor_orchestration") or "")
    pdf_filename = analysis.get("pdf_filename")

    st.markdown("**Proposed action**")
    with st.container(border=True):
        st.markdown(sections.get("recommendation")
                    or "Review the generated report and approve the recommended plan.")

    if sections.get("evidence"):
        st.markdown("**Supporting evidence**")
        with st.container(border=True):
            st.markdown(sections["evidence"][:1200])

    st.warning("Possible operational consequence: approving authorizes the "
               "recommended plan and notifications for this defect. The demo "
               "email step runs in dry-run mode.")

    reason = st.text_input("Optional note / rejection reason", key="approval_reason")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", type="primary", use_container_width=True, key="approve_btn"):
        if pdf_filename:
            record_report_decision(pdf_filename, "approved")
        st.session_state.approval_state = {"decision": "approved", "reason": reason}
        st.rerun()
    if col2.button("❌ Reject", use_container_width=True, key="reject_btn"):
        if pdf_filename:
            record_report_decision(pdf_filename, "rejected")
        st.session_state.approval_state = {"decision": "rejected", "reason": reason}
        st.rerun()


def render_approval_section(analysis):
    """Approval gate. Reads any prior decision from the persistent log."""
    pdf_filename = analysis.get("pdf_filename")
    prior = load_report_decisions().get(pdf_filename or "", {})
    decision = (st.session_state.get("approval_state") or {}).get("decision") or prior.get("decision")

    with st.container(border=True):
        st.markdown("#### Human approval")
        if decision == "approved":
            st.success("✅ This plan was **approved**.")
        elif decision == "rejected":
            st.error("❌ This plan was **rejected**.")
        else:
            st.warning("This plan is **awaiting human approval** before it is actioned.")
            if st.button("Review & approve →", type="primary", key="open_approval"):
                _approval_dialog(analysis)


# ---------------------------------------------------------------------------
# Section 4 — inspection tabs
# ---------------------------------------------------------------------------

def render_summary_tab(analysis):
    sections = _extract_report_sections(analysis.get("supervisor_orchestration") or "")
    events = st.session_state.get("agent_event_log") or []
    st.markdown("**Conclusion**")
    st.markdown(sections.get("root_cause") or sections.get("conclusion")
                or analysis.get("supervisor_orchestration") or "_No conclusion._")
    if sections.get("evidence"):
        st.markdown("**Key evidence**")
        st.markdown(sections["evidence"])
    if sections.get("recommendation"):
        st.markdown("**Recommended action**")
        st.markdown(sections["recommendation"])
    confidence = _confidence_from_events(events)
    st.markdown(f"**Confidence:** {f'{confidence*100:.0f}%' if confidence is not None else 'Not scored'}")
    if sections.get("limitations"):
        st.markdown("**Limitations**")
        st.markdown(sections["limitations"])


def render_evidence_tab(analysis):
    st.caption("Evidence drawn from the MES database during and around this "
               "investigation, grouped by source.")
    defect_type = analysis.get("defect_type")

    st.markdown("**Defect records**")
    if defect_type:
        render_defect_preview(defect_type)

    st.markdown("**Production trends**")
    render_overview_dashboard()

    events = st.session_state.get("agent_event_log") or []
    queried = _records_examined(events)
    st.caption(f"During this run the agents examined {queried:,} records across "
               f"{_tool_calls(events)} tool calls — see the SQL & Tools tab for each query.")


def render_sql_tools_tab(events):
    """One collapsed expander per tool call. SQL never expanded by default."""
    completed = {}
    sql_by_tool = {}
    tool_starts = []
    for e in events:
        etype = e.get("type")
        payload = e.get("payload") or {}
        if etype == "tool_started":
            name = str(payload.get("tool_name", ""))
            if not name.startswith("call_"):
                tool_starts.append(e)
        elif etype == "tool_completed":
            completed[payload.get("tool_use_id")] = e
        elif etype in ("sql_executed", "sql_failed") and payload.get("tool_use_id"):
            sql_by_tool.setdefault(payload["tool_use_id"], []).append(e)

    if not tool_starts:
        st.info("No tool calls were recorded for this run.")
        return

    for e in tool_starts:
        payload = e.get("payload") or {}
        name = payload.get("tool_name")
        tuid = payload.get("tool_use_id")
        done = completed.get(tuid, {})
        done_payload = done.get("payload") or {}
        sql_events = sql_by_tool.get(tuid, [])
        rows = None
        exec_ms = None
        for se in sql_events:
            sp = se.get("payload") or {}
            rows = sp.get("row_count") if rows is None else rows
            exec_ms = sp.get("execution_time_ms") if exec_ms is None else exec_ms
        ok = done_payload.get("status") != "error"
        mark = "✓" if ok else "✕"
        bits = [f"{mark} {name}"]
        if rows is not None:
            bits.append(f"{rows} rows")
        duration = done_payload.get("duration_ms")
        if duration is not None:
            bits.append(f"{duration} ms")
        with st.expander(" — ".join(bits)):
            args = payload.get("arguments")
            st.markdown(f"**Tool:** `{name}`")
            if not ok and done_payload.get("error"):
                st.error(done_payload["error"])
            if args:
                st.markdown("**Arguments**")
                st.json(args) if isinstance(args, (dict, list)) else st.write(args)
            for se in sql_events:
                sp = se.get("payload") or {}
                if sp.get("purpose"):
                    st.caption(f"Purpose: {sp['purpose']}")
                st.code(sp.get("sql", ""), language="sql")
                st.caption(f"Params: {sp.get('params') or '—'} · "
                           f"{sp.get('row_count')} rows · {sp.get('execution_time_ms')} ms "
                           f"· read-only validated")


def render_agent_trace_tab(events):
    """One collapsed expander per agent, chronologically. Raw JSON is tucked
    into a single final expander."""
    run, timeline = _group_events(events)
    activations = [entry for kind, entry in timeline if kind == "activation"]
    if not activations:
        st.info("No agent activity was recorded for this run.")
    for entry in activations:
        agent = _plain_agent_name(entry.get("agent_name"))
        status = entry.get("status")
        mark = "✓" if status == "completed" else ("✕" if status == "failed" else "…")
        duration = entry.get("duration_s")
        label = f"{mark} {agent}" + (f" — {duration} seconds" if duration else "")
        with st.expander(label):
            if entry.get("prompt"):
                st.markdown("**Goal / input received**")
                st.markdown("> " + str(entry["prompt"]).strip().replace("\n", "\n> "))
            rows = _records_examined(entry.get("items", []))
            tools = _tool_calls(entry.get("items", []))
            st.markdown(f"**Tools called:** {tools} · **records examined:** {rows:,}")
            if entry.get("result_preview"):
                st.markdown("**Output / conclusion**")
                st.markdown(entry["result_preview"])
            if entry.get("error"):
                st.error(entry["error"])
            footer = _metrics_caption(entry.get("metrics"))
            if footer:
                st.caption(footer)

    with st.expander("Raw event data"):
        st.json(events)


def render_report_tab(analysis):
    pdf_filename = analysis.get("pdf_filename")
    pdf_path = REPORTS_DIR / pdf_filename if pdf_filename else None

    if pdf_path and pdf_path.exists():
        st.success(f"Report generated: {pdf_filename}")
        markdown_text = analysis.get("supervisor_orchestration") or ""
        preview = "\n".join(markdown_text.splitlines()[:12]).strip()
        if preview:
            st.markdown("**Preview**")
            with st.container(border=True):
                st.markdown(preview)
        with open(pdf_path, "rb") as handle:
            pdf_bytes = handle.read()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download PDF Report", data=pdf_bytes,
                               file_name=pdf_filename, mime="application/pdf",
                               use_container_width=True, key="report_download")
        with col2:
            if st.button("👁️ View report", use_container_width=True, key="report_view"):
                st.query_params["pdf"] = pdf_filename
                st.rerun()
    else:
        st.error("The report file could not be found. Re-run the investigation to regenerate it.")


def render_inspection_tabs(analysis, events):
    summary_tab, evidence_tab, sql_tab, trace_tab, report_tab = st.tabs(
        ["Summary", "Evidence", "SQL & Tools", "Agent Trace", "Report"])
    with summary_tab:
        render_summary_tab(analysis)
    with evidence_tab:
        render_evidence_tab(analysis)
    with sql_tab:
        render_sql_tools_tab(events)
    with trace_tab:
        render_agent_trace_tab(events)
    with report_tab:
        render_report_tab(analysis)


# ---------------------------------------------------------------------------
# Empty and error states
# ---------------------------------------------------------------------------

def render_empty_state():
    with st.container(border=True):
        st.markdown("#### New here? It takes three steps")
        st.markdown(
            "1. **Choose** a defect type above.\n"
            "2. **Run** the investigation and watch the agents work.\n"
            "3. **Read** the conclusion, inspect the evidence, then approve and download the report."
        )
        st.caption("Sample defects to try: " + " · ".join(SAMPLE_INVESTIGATIONS))


def render_error_state():
    with st.container(border=True):
        st.error(st.session_state.get("run_error")
                 or "Something went wrong during the investigation.")
        col1, col2 = st.columns(2)
        if col1.button("Retry", type="primary", use_container_width=True, key="retry_run"):
            st.session_state.pop("run_error", None)
            if st.session_state.get("selected_defect") and st.session_state.get("run_params"):
                st.session_state.analysis_running = True
                st.session_state.work_pending = True
            st.rerun()
        if col2.button("Reset", use_container_width=True, key="reset_after_error"):
            for key in ["run_error", "agent_event_log", "analysis_running", "work_pending",
                        "analysis_started", "run_params"]:
                st.session_state.pop(key, None)
            st.session_state.current_analysis = {}
            st.rerun()
        with st.expander("Technical details"):
            st.write(st.session_state.get("run_error"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_investigate_page():
    st.markdown("## 🔎 Investigate")

    render_investigation_form()

    # Execute a pending run under the one-shot guard (agents are not
    # re-entrant — this preserves the existing non-reentrancy pattern).
    if st.session_state.get("analysis_running") and st.session_state.get("work_pending"):
        st.session_state.work_pending = False
        params = st.session_state.get("run_params", {})
        st.markdown("### 2 · Agent workflow")
        try:
            run_defect_analysis(
                defect_type=st.session_state.selected_defect,
                days_back=params.get("days_back", 7),
                include_oee=params.get("include_oee", False),
                include_downtime=params.get("include_downtime", False),
                include_changeover=params.get("include_changeover", False),
                include_maintenance=params.get("include_maintenance", True),
                render_fn=render_live_workflow,
            )
        finally:
            st.session_state.analysis_running = False
            st.rerun()
        return

    events = st.session_state.get("agent_event_log") or []

    if st.session_state.get("run_error"):
        st.markdown("### 2 · Agent workflow")
        if events:
            render_live_workflow(events, running=False)
        render_error_state()
        return

    if st.session_state.get("analysis_started") and st.session_state.get("current_analysis"):
        analysis = st.session_state.current_analysis
        st.markdown("### 2 · Agent workflow")
        render_live_workflow(events, running=False)
        st.markdown("### 3 · Investigation result")
        render_result_summary(analysis)
        render_approval_section(analysis)
        st.markdown("### 4 · Inspect the investigation")
        render_inspection_tabs(analysis, events)
    elif not st.session_state.get("analysis_running"):
        render_empty_state()


def _load_run_history():
    """Read persisted runs from runs/<id>/ for the history table."""
    rows = []
    if not RUNS_DIR.exists():
        return rows
    for run_dir in sorted(RUNS_DIR.glob("*/"), reverse=True):
        params_file = run_dir / "params.json"
        if not params_file.exists():
            continue
        try:
            params = json.loads(params_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        status, runtime, root_cause = "unknown", None, "—"
        events_file = run_dir / "events.jsonl"
        if events_file.exists():
            try:
                events = [json.loads(line) for line in
                          events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                run, _ = _group_events(events)
                if run.get("completed"):
                    status = "completed"
                    runtime = run["completed"]["payload"].get("duration_s")
                elif run.get("failed"):
                    status = "failed"
            except Exception:
                pass
        rows.append({
            "Status": status,
            "Investigation": params.get("defect_type", "—"),
            "Machine": "—",
            "Root cause": root_cause,
            "Confidence": "—",
            "Started": params.get("started", "—"),
            "Runtime": f"{runtime:.1f}s" if isinstance(runtime, (int, float)) else "—",
            "_dir": str(run_dir),
        })
    return rows


def render_history_page():
    st.markdown("## 🗂 Run History")
    rows = _load_run_history()
    if not rows:
        st.info("No previous runs found yet. Completed investigations will appear here.")
        return

    df = pd.DataFrame(rows)

    f1, f2, f3 = st.columns(3)
    with f1:
        statuses = ["All"] + sorted(df["Status"].unique().tolist())
        status_filter = st.selectbox("Status", statuses, key="hist_status")
    with f2:
        machines = ["All"] + sorted(df["Machine"].unique().tolist())
        machine_filter = st.selectbox("Machine", machines, key="hist_machine")
    with f3:
        date_filter = st.text_input("Started on/after (YYYY-MM-DD)", key="hist_date")

    view = df.copy()
    if status_filter != "All":
        view = view[view["Status"] == status_filter]
    if machine_filter != "All":
        view = view[view["Machine"] == machine_filter]
    if date_filter.strip():
        view = view[view["Started"].astype(str) >= date_filter.strip()]

    display_cols = ["Status", "Investigation", "Machine", "Root cause",
                    "Confidence", "Started", "Runtime"]
    selection = st.dataframe(
        view[display_cols], use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="history_table",
    )

    selected_rows = selection.get("selection", {}).get("rows", []) if selection else []
    if selected_rows:
        record = view.iloc[selected_rows[0]]
        st.session_state.selected_history_row = record.get("_dir")
        with st.container(border=True):
            st.markdown(f"**{record['Investigation']}** — {record['Status']}")
            st.caption(f"Started {record['Started']} · runtime {record['Runtime']}")
            st.caption(f"Run folder: `{record['_dir']}`")


def render_learning_page():
    st.markdown("## 📘 How It Works")
    st.markdown(
        "A **Supervisor** agent orchestrates specialist agents to investigate a "
        "manufacturing defect, grounding every claim in read-only database "
        "queries, then a **Verifier** checks the findings before a human approves "
        "the plan and a report is produced."
    )

    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#EFF4FF", color="#2563EB", fontname="sans-serif"];
        User -> Supervisor;
        Supervisor -> "Specialist Agents";
        "Specialist Agents" -> Verifier;
        Verifier -> "Human Approval";
        "Human Approval" -> "Final Report";
    }
    """)

    points = [
        ("Supervisor orchestration",
         "The Supervisor plans the workflow and delegates to specialist agents, one phase at a time."),
        ("Read-only tools",
         "Agents query the MES database through parameterized, read-only tools — they cannot modify data."),
        ("Visible SQL",
         "Every database query, its parameters, row counts and timing are shown in the SQL & Tools tab."),
        ("Evidence grounding",
         "Counts and correlations come from SQL, not from the model, so conclusions trace to real records."),
        ("Verification",
         "A Verifier agent reviews the findings before they are presented as conclusions."),
        ("Human approval",
         "A person approves or rejects the recommended plan; the decision is recorded in the report log."),
    ]
    for title, body in points:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(body)
            if st.session_state.get("learning_mode"):
                st.caption("Why this matters: it keeps the investigation auditable — "
                           "a reader can trace every conclusion back to the data and the human sign-off.")


def render_shared_pdf(pdf_path):
    """Standalone viewer for a report opened via a shared ?pdf= link."""
    st.markdown(f"## 📄 {pdf_path.name}")
    decision = load_report_decisions().get(pdf_path.name)
    if decision:
        verdict = "approved ✅" if decision.get("decision") == "approved" else "rejected ❌"
        st.info(f"Human review: this plan was **{verdict}** on {decision.get('at')}.")
    col1, col2, col3 = st.columns([2, 2, 1])
    if col1.button("✅ Approve plan", key="shared_approve"):
        record_report_decision(pdf_path.name, "approved")
        st.rerun()
    if col2.button("❌ Reject plan", key="shared_reject"):
        record_report_decision(pdf_path.name, "rejected")
        st.rerun()
    if col3.button("✖ Close", key="shared_close"):
        st.query_params.clear()
        st.rerun()
    pdf_data = display_pdf_viewer(pdf_path)
    if pdf_data:
        st.download_button("📥 Download This Report", data=pdf_data,
                           file_name=pdf_path.name, mime="application/pdf",
                           key="shared_download")


def main():
    """Guided investigation workspace."""
    st.set_page_config(
        page_title="MES Quality Investigator",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Shared-link report viewer works from any entry point.
    url_pdf = get_pdf_from_url()
    if url_pdf is not None:
        render_sidebar()
        render_shared_pdf(url_pdf)
        return

    render_sidebar()
    pages = [
        st.Page(render_investigate_page, title="Investigate", icon="🔎", default=True),
        st.Page(render_history_page, title="Run History", icon="🗂"),
        st.Page(render_learning_page, title="How It Works", icon="📘"),
    ]
    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()

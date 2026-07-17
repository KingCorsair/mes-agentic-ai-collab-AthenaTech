"""
Strands agents for MES  application
Contains Monitor, Analyzer, Planner, and Verifier agents for manufacturing quality analysis
"""

import re
import itertools
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent, HookProvider
from strands.models.anthropic import AnthropicModel

load_dotenv(Path(__file__).parent / ".env")
# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Setup logging
def setup_logging():
    log_level = os.getenv('MES_LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('MES_LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Use one consistent reports directory beside strands_agent.py.
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _md_inline(text):
    """Escape XML-unsafe chars, then convert inline markdown to ReportLab tags."""
    text = str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    return text

def _markdown_to_flowables(text, styles):
    """Convert a block of markdown-ish agent text into ReportLab flowables."""
    flowables = []
    bullet_style = ParagraphStyle('MDBullet', parent=styles['Normal'],
                                  leftIndent=18, bulletIndent=6, spaceAfter=4)
    lines = str(text).split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if re.fullmatch(r'[=\-_*]{3,}', line):
            flowables.append(Spacer(1, 8))
            continue
        if line.startswith('|') and line.endswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip())
                i += 1
            data = []
            for r in rows:
                cells = [c.strip() for c in r.strip('|').split('|')]
                if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                    continue
                data.append([Paragraph(_md_inline(c), styles['Normal'])
                             for c in cells])
            if data:
                tbl = Table(data, hAlign='LEFT')
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe5f1')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                ]))
                flowables.append(tbl)
                flowables.append(Spacer(1, 10))
            continue
        m = re.match(r'(#{1,4})\s+(.*)', line)
        if m:
            level = min(len(m.group(1)), 3)
            flowables.append(Paragraph(_md_inline(m.group(2)),
                                       styles[f'Heading{level}']))
            flowables.append(Spacer(1, 6))
            continue
        m = re.fullmatch(r'\*\*(.+?)\*\*:?', line)
        if m:
            flowables.append(Paragraph(f'<b>{_md_inline(m.group(1))}</b>',
                                       styles['Heading3']))
            flowables.append(Spacer(1, 4))
            continue
        m = re.match(r'[-*\u2022]\s+(.*)', line)
        if m:
            flowables.append(Paragraph(_md_inline(m.group(1)), bullet_style,
                                       bulletText='\u2022'))
            continue
        m = re.match(r'(\d+)[.)]\s+(.*)', line)
        if m:
            flowables.append(Paragraph(_md_inline(m.group(2)), bullet_style,
                                       bulletText=f'{m.group(1)}.'))
            continue
        flowables.append(Paragraph(_md_inline(line), styles['Normal']))
        flowables.append(Spacer(1, 6))
    return flowables


def render_markdown_report_pdf(markdown_text, filename=None):
    """Render the Supervisor's final markdown report as a PDF."""

    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "ReportLab is not installed in the Python environment "
            "running Streamlit. Install it with: python -m pip install reportlab"
        )

    report_text = str(markdown_text).strip()

    if not report_text:
        raise ValueError("Cannot generate a PDF from an empty report.")

    if filename is None:
        filename = (
            f"MES_Final_Report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Avoid creating report.pdf.pdf.
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]

    filepath = REPORTS_DIR / f"{filename}.pdf"

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=TA_CENTER,
    )

    story = [
        Paragraph(
            "Manufacturing Execution System Analysis Report",
            title_style,
        ),
        Spacer(1, 20),
        Paragraph(
            (
                "Generated: "
                f"{datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
            ),
            styles["Normal"],
        ),
        Spacer(1, 20),
    ]

    story.extend(
        _markdown_to_flowables(
            report_text,
            styles,
        )
    )

    doc.build(story)

    if not filepath.exists():
        raise RuntimeError(
            f"PDF generation finished, but the file was not created: "
            f"{filepath.resolve()}"
        )

    if filepath.stat().st_size == 0:
        raise RuntimeError(
            f"The generated PDF is empty: {filepath.resolve()}"
        )

    logger.info(
        "PDF successfully generated: %s (%s bytes)",
        filepath.resolve(),
        filepath.stat().st_size,
    )

    return str(filepath.resolve())


class _ObservabilityHooks(HookProvider):
    """Strands hook provider shared by all six agents.

    Emits tool_started / tool_completed events through the manager's
    _emit so the UI can show every tool call any agent makes — the tool
    name, the arguments the model chose, and what came back.
    """

    def __init__(self, manager):
        self._manager = manager
        # toolUseId -> start time, for per-call durations. Tool calls can
        # run on SDK worker threads, so keep this a plain dict keyed by
        # the unique toolUseId rather than assuming call order.
        self._tool_starts = {}

    def register_hooks(self, registry, **kwargs):
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _before_tool(self, event):
        tool_use = event.tool_use or {}
        tool_use_id = tool_use.get("toolUseId")
        self._tool_starts[tool_use_id] = time.time()
        self._manager._emit("tool_started", {
            "tool_name": tool_use.get("name"),
            "tool_use_id": tool_use_id,
            "arguments": self._manager._preview(tool_use.get("input"), 300),
        })

    def _after_tool(self, event):
        tool_use = event.tool_use or {}
        tool_use_id = tool_use.get("toolUseId")
        started = self._tool_starts.pop(tool_use_id, None)
        payload = {
            "tool_name": tool_use.get("name"),
            "tool_use_id": tool_use_id,
            "duration_ms": round((time.time() - started) * 1000) if started else None,
        }
        if event.exception is not None:
            payload["status"] = "error"
            payload["error"] = str(event.exception)
        else:
            result = event.result or {}
            payload["status"] = result.get("status", "success") if isinstance(result, dict) else "success"
            payload["result_preview"] = self._manager._preview(
                result.get("content", result) if isinstance(result, dict) else result, 400)
        self._manager._emit("tool_completed", payload)


class MESAgentManager:
    """Manager class for MES agents focused on manufacturing quality analysis"""
    
    def __init__(self, db_path: str = None, model_id: str = None, region_name: str = None,
                 on_event=None):
        """Initialize the MES Agent Manager"""

        # Optional observer for the UI: receives structured event dicts
        # describing everything the agentic system does (see _emit).
        self.on_event = on_event
        self._active_agent = None
        self.event_log = []
        self._event_seq = itertools.count(1)
        self._observability_hooks = _ObservabilityHooks(self)

        # Get parameters from environment variables with fallbacks
        if db_path is None:
            db_path = os.getenv('MES_DB_PATH')
            if db_path is None:
                proj_dir = os.path.abspath('')
                db_path = os.path.join(proj_dir, 'mes.db')
        
        if model_id is None:
            model_id = os.getenv("MES_MODEL_ID", "claude-sonnet-4-6")
        
        if region_name is None:
            region_name = os.getenv('AWS_REGION', 'us-west-2')
        
        # Email configuration from environment variables
        self.sender_email = os.getenv('MES_SENDER_EMAIL', 'operations.team@example.com')
        self.recipient_email = os.getenv('MES_RECIPIENT_EMAIL', 'operations.team@example.com')
        self.base_url = os.getenv('MES_BASE_URL', 'https://df4n.cloudfront.net/proxy/8501')
        
        # Database path
        self.db_path = db_path

        # Anthropic API key from .env / environment
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")

        self._model_id = model_id

        self._model_client_args = {
            "api_key": api_key,
            "timeout": float(os.getenv("MES_API_TIMEOUT", "40")),
            "max_retries": int(os.getenv("MES_API_RETRIES", "2")),
        }

        self._model_max_tokens = int(
            os.getenv("MES_MAX_TOKENS", "8296")
        )

        self._model_params = {
            "temperature": float(
                os.getenv("MES_TEMPERATURE", "0.2")
            ),
        }

        self.region_name = region_name
        
        # Define allowed table names for security
        self.allowed_tables = {
            'OEEMetrics', 'Machines', 'WorkCenters', 'Downtimes', 'WorkOrders', 
            'Products', 'Shifts', 'Employees', 'Defects', 'QualityControl'
        }
        
        # Log configuration
        logger.info(f"MES Agent Manager initialized with:")
        logger.info(f"  Database Path: {self.db_path}")
        logger.info(f"  Model ID: {model_id}")
        logger.info(f"  AWS Region: {region_name}")
        logger.info(f"  Sender Email: {self.sender_email}")
        logger.info(f"  Recipient Email: {self.recipient_email}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Max Tokens: {os.getenv('MES_MAX_TOKENS', '8296')}")
        logger.info(f"  Temperature: {os.getenv('MES_TEMPERATURE', '0.2')}")
        
        # Initialize tools and agents
        self._init_database_tools()
        self._init_email_tools()
        self._init_monitor_tools()
        self._init_analyzer_tools()
        self._init_planner_tools()
        self._init_executor_tools()
        self._init_verifier_tools()
        self._init_agents()
        self._init_supervisor_agent()
    
    def _create_model(self):
        """Create an independent model client for one agent."""

        return AnthropicModel(
            client_args=dict(self._model_client_args),
            model_id=self._model_id,
            max_tokens=self._model_max_tokens,
            params=dict(self._model_params),
        )

    def _emit(self, event_type: str, payload: dict = None):
        """Record one observability event and forward it to the UI callback.

        Called from SDK worker threads as well as the main thread, so it
        must never touch Streamlit itself — the callback owns delivery
        (e.g. via a thread-safe queue). A failing callback must never
        break an analysis run.
        """
        event = {
            "seq": next(self._event_seq),
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": event_type,
            "agent": self._active_agent or "supervisor",
            "payload": payload or {},
        }
        self.event_log.append(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as e:
                logger.warning(f"on_event callback failed for {event_type}: {e}")
        return event

    @staticmethod
    def _preview(value, limit: int = 300) -> str:
        """Whitespace-collapsed, length-capped repr for event payloads."""
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[:limit] + " …[truncated]"

    def _metrics_snapshot(self, agent_obj) -> dict:
        """Cycle/token/latency summary from an agent's event loop metrics.

        Values are cumulative over the agent object's lifetime; agents are
        rebuilt per manager, so per-run they are effectively per-agent totals.
        """
        try:
            metrics = getattr(agent_obj, "event_loop_metrics", None)
            if metrics is None or not hasattr(metrics, "get_summary"):
                return {}
            summary = metrics.get_summary()
            usage = summary.get("accumulated_usage", {}) or {}
            return {
                "cycles": summary.get("total_cycles"),
                "model_seconds": round(summary.get("total_duration") or 0, 1),
                "input_tokens": usage.get("inputTokens"),
                "output_tokens": usage.get("outputTokens"),
            }
        except Exception as e:
            logger.warning(f"Metrics snapshot failed: {e}")
            return {}

    def _save_event_log(self):
        """Persist the run's event log next to the other run artifacts."""
        try:
            run_dir = getattr(self, "current_run_dir", None)
            if run_dir is None or not self.event_log:
                return
            lines = "\n".join(json.dumps(e, default=str) for e in self.event_log)
            (run_dir / "events.jsonl").write_text(lines + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning(f"Event log persistence failed: {e}")

    def get_db_connection(self):
        """Get a database connection"""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database file not found: {self.db_path}")
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        return sqlite3.connect(self.db_path)
    
    def _validate_table_name(self, table_name: str) -> bool:
        """Validate table name against allowed list"""
        return table_name in self.allowed_tables
    
    def _execute_safe_query(self, query: str, params: tuple = None):
        """Execute SQL query safely with parameterized queries"""
        logger.info(f"Executing parameterized SQL query")
        logger.debug(f"SQL: {' '.join(str(query).split())[:300]}")
        logger.debug(f"SQL params: {params}")
        start_time = time.time()
        
        try:
            conn = self.get_db_connection()
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Process datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        if df[col].str.contains('-').any() and df[col].str.contains(':').any():
                            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
            
            # Round float columns
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = df[col].round(2)
            
            result = {
                "success": True,
                "rows": df.to_dict(orient="records"),
                "column_names": df.columns.tolist(),
                "row_count": len(df),
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "dataframe": df
            }

            logger.info(f"Query executed successfully: {len(df)} rows returned")
            self._emit("sql_executed", {
                "sql": self._preview(query, 600),
                "params": self._preview(params, 200) if params else None,
                "row_count": len(df),
                "execution_time_ms": result["execution_time_ms"],
            })
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing SQL query: {error_msg}")
            error_result = {
                "success": False,
                "error": error_msg,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            self._emit("sql_failed", {
                "sql": self._preview(query, 600),
                "params": self._preview(params, 200) if params else None,
                "error": error_msg,
                "execution_time_ms": error_result["execution_time_ms"],
            })
            return error_result
    
    def _init_database_tools(self):
        """Initialize core database tools"""
        
        @tool
        def execute_sql(sql_query: str):
            """Execute predefined SQL queries against the MES database - only allows specific safe queries"""
            logger.info(f"Executing predefined SQL query")
            
            # Define allowed safe queries with parameterized structure
            allowed_queries = {
                "get_tables": "SELECT name FROM sqlite_master WHERE type='table'",
                "get_recent_oee": """
                    SELECT 
                        oee.Date,
                        m.Name as MachineName,
                        m.Type as MachineType,
                        wc.Name as WorkCenterName,
                        oee.Availability,
                        oee.Performance,
                        oee.Quality,
                        oee.OEE
                    FROM 
                        OEEMetrics oee
                    JOIN 
                        Machines m ON oee.MachineID = m.MachineID
                    JOIN 
                        WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
                    ORDER BY 
                        oee.Date DESC
                    LIMIT 100
                """,
                "get_recent_downtime": """
                    SELECT 
                        dt.StartTime,
                        dt.EndTime,
                        dt.Duration,
                        dt.Reason,
                        m.Name as MachineName,
                        m.Type as MachineType
                    FROM 
                        Downtimes dt
                    JOIN 
                        Machines m ON dt.MachineID = m.MachineID
                    ORDER BY 
                        dt.StartTime DESC
                    LIMIT 100
                """
            }
            
            # Check if the query is in allowed list
            query_key = sql_query.strip().lower()
            if query_key in allowed_queries:
                return self._execute_safe_query(allowed_queries[query_key])
            else:
                # For security, only allow predefined queries
                logger.warning(f"Query not in allowed list: {sql_query}")
                self._emit("guardrail_triggered", {
                    "guardrail": "sql_allowlist",
                    "attempted_query": self._preview(sql_query, 400),
                    "outcome": "rejected - only predefined safe queries are allowed",
                })
                return {
                    "success": False,
                    "error": "Only predefined safe queries are allowed for security reasons"
                }
        
        self.execute_sql_tool = execute_sql

    def _init_email_tools(self):
        """Initialize Email tools"""
        
        @tool
        def send_email(subject: str, email_body: str, pdf_filename: str = None):
            """Send email for the short term action items with PDF link"""
            if os.getenv("MES_EMAIL_DRY_RUN", "true").lower() == "true":
                return {
                    "success": True,
                    "message": "Dry run: email not sent",
                    "subject": subject,
                    "body": email_body,
                    "pdf_filename": pdf_filename,
                }

            logger.info(f"Sending email with following detail: subject - {subject}")
            start_time = time.time()
            
            # Create SES client using IAM role
            ses_client = boto3.client('ses', region_name=self.region_name)

            # Email parameters from environment variables
            SENDER = self.sender_email
            RECIPIENT = self.recipient_email
            SUBJECT = subject
            
            # Add PDF link to email body if filename is provided
            if pdf_filename:
                pdf_link = f"{self.base_url}/pdf={pdf_filename}?pdf={pdf_filename}"
                email_body += f"\n\nDetailed PDF Report: {pdf_link}"
            
            # Email content
            BODY_TEXT = email_body
            BODY_HTML = f"""
            <html>
            <body>
                <h1>MES Execution Plan</h1>
                <p>{email_body.replace(chr(10), '<br>')}</p>
            </body>
            </html>
            """
            
            try:
                response = ses_client.send_email(
                    Destination={
                        'ToAddresses': [RECIPIENT]
                    },
                    Message={
                        'Body': {
                            'Html': {
                                'Charset': 'UTF-8',
                                'Data': BODY_HTML
                            },
                            'Text': {
                                'Charset': 'UTF-8',
                                'Data': BODY_TEXT
                            }
                        },
                        'Subject': {
                            'Charset': 'UTF-8',
                            'Data': SUBJECT
                        }
                    },
                    Source=SENDER
                )
                logger.info(f"Email sent! Message ID: {response['MessageId']}")
                
                result = {
                    "success": True,
                    "message": f"Email sent! Message ID: {response['MessageId']}",
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                return result
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error sending email: {error_msg}")
                error_result = {
                    "success": False,
                    "error": error_msg,
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                return error_result
        
        self.execute_email_send = send_email

    def _init_monitor_tools(self):
        """Initialize Monitor Agent tools - Captures & contextualizes operational data"""
        @tool
        def fetch_defect_records(defect_type: str, days_back: int = 7):
            """Fetch individual defect occurrences for ONE defect type from the
            Defects table, with timestamps and full context. Returns one row per
            occurrence: check date/time, severity, quantity, location, recorded
            root cause, action taken, plus the product, machine, work center,
            operator, and shift involved. Use this for defect timelines and
            correlating defect timing against maintenance or downtime events.
            Newest first, capped at 200 rows."""
            # Validate inputs (match the house style of the other tools)
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")

            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            query = """
            SELECT
                qc.Date as CheckDate,
                d.DefectType,
                d.Severity,
                d.Quantity as DefectQuantity,
                d.Location,
                d.RootCause,
                d.ActionTaken,
                p.Name as ProductName,
                m.Name as MachineName,
                wc.Name as WorkCenterName,
                e.Name as OperatorName,
                s.Name as ShiftName,
                wo.OrderID
            FROM
                Defects d
            JOIN
                QualityControl qc ON d.CheckID = qc.CheckID
            JOIN
                WorkOrders wo ON qc.OrderID = wo.OrderID
            JOIN
                Products p ON wo.ProductID = p.ProductID
            JOIN
                Machines m ON wo.MachineID = m.MachineID
            JOIN
                WorkCenters wc ON wo.WorkCenterID = wc.WorkCenterID
            JOIN
                Employees e ON wo.EmployeeID = e.EmployeeID
            JOIN
                Shifts s ON e.ShiftID = s.ShiftID
            WHERE
                d.DefectType = ?
                AND date(qc.Date) >= ?
            ORDER BY
                qc.Date DESC
            LIMIT 200
            """

            return self._execute_safe_query(query, (defect_type, cutoff_date))
        
        @tool
        def fetch_oee_metrics(days_back: int = 7):
            """Fetch OEE metrics and identify drops in performance"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                oee.Date,
                m.Name as MachineName,
                m.Type as MachineType,
                wc.Name as WorkCenterName,
                oee.Availability,
                oee.Performance,
                oee.Quality,
                oee.OEE,
                CASE 
                    WHEN oee.OEE < 0.6 THEN 'Critical'
                    WHEN oee.OEE < 0.75 THEN 'Low'
                    ELSE 'Acceptable'
                END as OEEStatus
            FROM 
                OEEMetrics oee
            JOIN 
                Machines m ON oee.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
            WHERE 
                oee.Date >= ?
            ORDER BY 
                oee.OEE ASC, oee.Date DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        @tool
        def fetch_downtime_events(days_back: int = 7):
            """Fetch downtime events and line stoppages"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                dt.StartTime,
                dt.EndTime,
                dt.Duration,
                dt.Reason,
                m.Name as MachineName,
                m.Type as MachineType,
                wc.Name as WorkCenterName,
                wo.OrderID,
                p.Name as ProductName,
                s.Name as ShiftName,
                e.Name as OperatorName
            FROM 
                Downtimes dt
            JOIN 
                Machines m ON dt.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
            LEFT JOIN 
                WorkOrders wo ON m.MachineID = wo.MachineID
                AND dt.StartTime BETWEEN wo.ActualStartTime AND wo.ActualEndTime
            LEFT JOIN 
                Products p ON wo.ProductID = p.ProductID
            LEFT JOIN 
                Employees e ON wo.EmployeeID = e.EmployeeID
            LEFT JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            WHERE 
                date(dt.StartTime) >= ?
            ORDER BY 
                dt.Duration DESC, dt.StartTime DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        @tool
        def fetch_historical_patterns(days_back: int = 7):
            """Fetch historical stoppage patterns and context"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                date(dt.StartTime) as StoppageDate,
                strftime('%w', dt.StartTime) as DayOfWeek,
                strftime('%H', dt.StartTime) as HourOfDay,
                dt.Reason,
                COUNT(*) as EventCount,
                AVG(dt.Duration) as AvgDuration,
                m.Type as MachineType,
                wc.Name as WorkCenterName,
                s.Name as ShiftName
            FROM 
                Downtimes dt
            JOIN 
                Machines m ON dt.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
            LEFT JOIN 
                WorkOrders wo ON m.MachineID = wo.MachineID
                AND dt.StartTime BETWEEN wo.ActualStartTime AND wo.ActualEndTime
            LEFT JOIN
                Employees e ON wo.EmployeeID = e.EmployeeID
            LEFT JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            WHERE 
                date(dt.StartTime) >= ?
            GROUP BY 
                date(dt.StartTime), dt.Reason, m.Type, wc.Name, s.Name
            ORDER BY 
                EventCount DESC, AvgDuration DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        @tool
        def fetch_work_orders_context(days_back: int = 7):
            """Fetch work orders context and batch reports"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                wo.OrderID,
                wo.Status,
                wo.PlannedStartTime,
                wo.ActualStartTime,
                wo.PlannedEndTime,
                wo.ActualEndTime,
                wo.Quantity as PlannedQuantity,
                wo.ActualProduction,
                wo.Scrap,
                p.Name as ProductName,
                p.Category as ProductCategory,
                m.Name as MachineName,
                wc.Name as WorkCenterName,
                e.Name as OperatorName,
                s.Name as ShiftName,
                ROUND((wo.ActualProduction * 100.0 / wo.Quantity), 2) as CompletionRate
            FROM 
                WorkOrders wo
            JOIN 
                Products p ON wo.ProductID = p.ProductID
            JOIN 
                Machines m ON wo.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON wo.WorkCenterID = wc.WorkCenterID
            JOIN 
                Employees e ON wo.EmployeeID = e.EmployeeID
            JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            WHERE 
                date(wo.ActualStartTime) >= ?
            ORDER BY 
                wo.ActualStartTime DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        @tool
        def fetch_operator_logs(days_back: int = 7):
            """Fetch operator logs and shift performance"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                wo.ActualStartTime as WorkDate,
                e.Name as OperatorName,
                e.Role as OperatorRole,
                s.Name as ShiftName,
                COUNT(wo.OrderID) as OrdersHandled,
                AVG(wo.ActualProduction * 100.0 / wo.Quantity) as AvgCompletionRate,
                SUM(wo.Scrap) as TotalScrap,
                wc.Name as WorkCenterName,
                m.Type as MachineType
            FROM 
                WorkOrders wo
            JOIN 
                Employees e ON wo.EmployeeID = e.EmployeeID
            JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            JOIN 
                WorkCenters wc ON wo.WorkCenterID = wc.WorkCenterID
            JOIN 
                Machines m ON wo.MachineID = m.MachineID
            WHERE 
                date(wo.ActualStartTime) >= ?
            GROUP BY 
                date(wo.ActualStartTime), e.EmployeeID, s.ShiftID, wc.WorkCenterID
            ORDER BY 
                wo.ActualStartTime DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        self.monitor_tools = [
            fetch_oee_metrics,
            fetch_downtime_events,
            fetch_historical_patterns,
            fetch_work_orders_context,
            fetch_operator_logs,
            fetch_defect_records          
        ]
        

    def _init_analyzer_tools(self):
        """Initialize Analyzer Agent tools - Identifies root causes and performs reasoning"""
        
        @tool
        def analyze_downtime_correlations(days_back: int = 7):
            """Analyze correlations between downtime and specific factors"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                dt.Reason as DowntimeReason,
                s.Name as ShiftName,
                e.Name as OperatorName,
                p.Name as ProductName,
                m.Type as MachineType,
                COUNT(*) as EventCount,
                AVG(dt.Duration) as AvgDuration,
                SUM(dt.Duration) as TotalDuration,
                wc.Name as WorkCenterName
            FROM 
                Downtimes dt
            JOIN 
                Machines m ON dt.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
            LEFT JOIN 
                WorkOrders wo ON m.MachineID = wo.MachineID
                AND dt.StartTime BETWEEN wo.ActualStartTime AND wo.ActualEndTime
            LEFT JOIN 
                Products p ON wo.ProductID = p.ProductID
            LEFT JOIN 
                Employees e ON wo.EmployeeID = e.EmployeeID
            LEFT JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            WHERE 
                date(dt.StartTime) >= ?
            GROUP BY 
                dt.Reason, s.Name, e.Name, p.Name, m.Type
            ORDER BY 
                TotalDuration DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        @tool
        def analyze_batch_changeover_time(days_back: int = 7):
            """Analyze batch changeover times vs benchmarks"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            WITH changeover_times AS (
                SELECT 
                    wo1.OrderID as PrevOrder,
                    wo2.OrderID as NextOrder,
                    wo1.ProductID as PrevProduct,
                    wo2.ProductID as NextProduct,
                    wo1.MachineID,
                    (julianday(wo2.ActualStartTime) - julianday(wo1.ActualEndTime)) * 24 as ChangeoverHours,
                    m.Name as MachineName,
                    m.Type as MachineType,
                    p1.Name as PrevProductName,
                    p2.Name as NextProductName,
                    s.Name as ShiftName
                FROM 
                    WorkOrders wo1
                JOIN 
                    WorkOrders wo2 ON wo1.MachineID = wo2.MachineID 
                    AND wo2.ActualStartTime > wo1.ActualEndTime
                JOIN 
                    Machines m ON wo1.MachineID = m.MachineID
                JOIN 
                    Products p1 ON wo1.ProductID = p1.ProductID
                JOIN 
                    Products p2 ON wo2.ProductID = p2.ProductID
                LEFT JOIN
                    Employees e2 ON wo2.EmployeeID = e2.EmployeeID
                LEFT JOIN 
                    Shifts s ON e2.ShiftID = s.ShiftID
                WHERE 
                    date(wo1.ActualEndTime) >= ?
                    AND date(wo2.ActualStartTime) >= ?
                    AND (julianday(wo2.ActualStartTime) - julianday(wo1.ActualEndTime)) * 24 < 24
                    AND (julianday(wo2.ActualStartTime) - julianday(wo1.ActualEndTime)) * 24 > 0
            )
            SELECT 
                MachineType,
                MachineName,
                PrevProductName,
                NextProductName,
                ShiftName,
                COUNT(*) as ChangeoverCount,
                AVG(ChangeoverHours * 60) as AvgChangeoverMinutes,
                MIN(ChangeoverHours * 60) as MinChangeoverMinutes,
                MAX(ChangeoverHours * 60) as MaxChangeoverMinutes,
                CASE 
                    WHEN AVG(ChangeoverHours * 60) > 120 THEN 'Excessive'
                    WHEN AVG(ChangeoverHours * 60) > 60 THEN 'Above Benchmark'
                    ELSE 'Acceptable'
                END as ChangeoverStatus
            FROM 
                changeover_times
            GROUP BY 
                MachineType, MachineName, PrevProductName, NextProductName, ShiftName
            ORDER BY 
                AvgChangeoverMinutes DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date, cutoff_date))

        @tool
        def identify_performance_patterns(days_back: int = 7):
            """Identify patterns in machine and operator performance"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                m.Name as MachineName,
                m.Type as MachineType,
                e.Name as OperatorName,
                s.Name as ShiftName,
                wc.Name as WorkCenterName,
                COUNT(wo.OrderID) as TotalOrders,
                AVG(oee.OEE) as AvgOEE,
                AVG(oee.Availability) as AvgAvailability,
                AVG(oee.Performance) as AvgPerformance,
                AVG(oee.Quality) as AvgQuality,
                SUM(wo.Scrap) as TotalScrap,
                AVG(wo.ActualProduction * 100.0 / wo.Quantity) as AvgCompletionRate,
                COUNT(dt.Duration) as DowntimeEvents,
                SUM(dt.Duration) as TotalDowntime
            FROM 
                WorkOrders wo
            JOIN 
                Machines m ON wo.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON m.WorkCenterID = wc.WorkCenterID
            JOIN 
                Employees e ON wo.EmployeeID = e.EmployeeID
            JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            LEFT JOIN 
                OEEMetrics oee ON m.MachineID = oee.MachineID 
                AND date(oee.Date) = date(wo.ActualStartTime)
            LEFT JOIN 
                Downtimes dt ON m.MachineID = dt.MachineID 
                AND date(dt.StartTime) = date(wo.ActualStartTime)
            WHERE 
                date(wo.ActualStartTime) >= ?
            GROUP BY 
                m.MachineID, e.EmployeeID, s.ShiftID
            ORDER BY 
                AvgOEE ASC, TotalDowntime DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        @tool
        def analyze_quality_defects(days_back: int = 7):
            """Analyze quality defects and their root causes"""
            # Validate input
            days_back = int(days_back)
            if days_back < 0 or days_back > 3650:
                raise ValueError("days_back must be between 0 and 3650")
            
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                d.DefectType,
                d.Severity,
                d.Location,
                d.RootCause,
                d.ActionTaken,
                COUNT(*) as DefectCount,
                p.Name as ProductName,
                p.Category as ProductCategory,
                m.Name as MachineName,
                m.Type as MachineType,
                wc.Name as WorkCenterName,
                e.Name as OperatorName,
                s.Name as ShiftName,
                AVG(qc.DefectRate) as AvgDefectRate,
                AVG(qc.YieldRate) as AvgYieldRate
            FROM 
                Defects d
            JOIN 
                QualityControl qc ON d.CheckID = qc.CheckID
            JOIN 
                WorkOrders wo ON qc.OrderID = wo.OrderID
            JOIN 
                Products p ON wo.ProductID = p.ProductID
            JOIN 
                Machines m ON wo.MachineID = m.MachineID
            JOIN 
                WorkCenters wc ON wo.WorkCenterID = wc.WorkCenterID
            JOIN 
                Employees e ON wo.EmployeeID = e.EmployeeID
            JOIN 
                Shifts s ON e.ShiftID = s.ShiftID
            WHERE 
                date(qc.Date) >= ?
            GROUP BY 
                d.DefectType, d.RootCause, p.ProductID, m.MachineID, e.EmployeeID, s.ShiftID
            ORDER BY 
                DefectCount DESC, d.Severity DESC
            """
            
            return self._execute_safe_query(query, (cutoff_date,))

        self.analyzer_tools = [
            analyze_downtime_correlations,
            analyze_batch_changeover_time,
            identify_performance_patterns,
            analyze_quality_defects
        ]

    def _init_planner_tools(self):
        """Initialize Planner Agent tools - Suggests actionable plans and creates PDF reports"""
        
        @tool
        def create_action_plan(analysis_data: str, priority_level: str = "High"):
            """Create actionable improvement plan based on analysis"""
            action_plan = {
                "priority": priority_level,
                "timestamp": datetime.now().isoformat(),
                "analysis_summary": analysis_data[:500] + "..." if len(analysis_data) > 500 else analysis_data,
                "immediate_actions": [
                    "Review identified problem areas",
                    "Implement monitoring for critical metrics",
                    "Schedule maintenance for problem machines"
                ],
                "short_term_actions": [
                    "Standardize changeover procedures",
                    "Provide additional operator training",
                    "Optimize batch scheduling"
                ],
                "long_term_actions": [
                    "Invest in predictive maintenance systems",
                    "Upgrade critical equipment",
                    "Implement advanced quality control"
                ]
            }
            
            return action_plan


        @tool
        def generate_pdf_report(report_data: dict, filename: str = None):
            """Generate a PDF report containing findings and action plans."""

            if not REPORTLAB_AVAILABLE:
                return {
                    "success": False,
                    "error": (
                        "ReportLab is not installed in the Python environment "
                        "running Streamlit."
                    ),
                }

            if filename is None:
                filename = (
                    f"MES_Analysis_Report_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

            if filename.lower().endswith(".pdf"):
                filename = filename[:-4]

            pdf_filename = f"{filename}.pdf"
            filepath = REPORTS_DIR / pdf_filename

            try:
                doc = SimpleDocTemplate(
                    str(filepath),
                    pagesize=A4,
                )

                styles = getSampleStyleSheet()
                story = []

                title_style = ParagraphStyle(
                    "CustomTitle",
                    parent=styles["Title"],
                    fontSize=24,
                    spaceAfter=30,
                    textColor=colors.darkblue,
                    alignment=TA_CENTER,
                )

                story.append(
                    Paragraph(
                        "Manufacturing Execution System Analysis Report",
                        title_style,
                    )
                )

                story.append(Spacer(1, 30))

                story.append(
                    Paragraph(
                        (
                            "Generated: "
                            f"{datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
                        ),
                        styles["Normal"],
                    )
                )

                story.append(Spacer(1, 20))

                if "executive_summary" in report_data:
                    story.append(
                        Paragraph(
                            "Executive Summary",
                            styles["Heading1"],
                        )
                    )

                    story.extend(
                        _markdown_to_flowables(
                            report_data["executive_summary"],
                            styles,
                        )
                    )

                    story.append(PageBreak())

                for section, content in report_data.items():
                    if section == "executive_summary":
                        continue

                    section_title = section.replace("_", " ").title()

                    story.append(
                        Paragraph(
                            section_title,
                            styles["Heading1"],
                        )
                    )

                    story.append(Spacer(1, 12))

                    if isinstance(content, str):
                        story.extend(
                            _markdown_to_flowables(
                                content,
                                styles,
                            )
                        )

                    elif isinstance(content, list):
                        for item in content:
                            story.extend(
                                _markdown_to_flowables(
                                    f"- {item}",
                                    styles,
                                )
                            )

                    elif isinstance(content, dict):
                        for key, value in content.items():
                            key_formatted = key.replace("_", " ").title()

                            if isinstance(value, list):
                                story.append(
                                    Paragraph(
                                        _md_inline(key_formatted),
                                        styles["Heading2"],
                                    )
                                )

                                for item in value:
                                    story.append(
                                        Paragraph(
                                            _md_inline(item),
                                            styles["Normal"],
                                        )
                                    )

                                    story.append(Spacer(1, 6))

                            else:
                                story.append(
                                    Paragraph(
                                        (
                                            f"<b>{_md_inline(key_formatted)}:</b> "
                                            f"{_md_inline(value)}"
                                        ),
                                        styles["Normal"],
                                    )
                                )

                                story.append(Spacer(1, 6))

                    else:
                        story.extend(
                            _markdown_to_flowables(
                                str(content),
                                styles,
                            )
                        )

                    story.append(Spacer(1, 20))

                story.append(Spacer(1, 30))

                story.append(
                    Paragraph(
                        (
                            "Report generated on "
                            f"{datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
                        ),
                        styles["Normal"],
                    )
                )

                doc.build(story)

                if not filepath.exists():
                    raise RuntimeError(
                        f"PDF file was not created: {filepath.resolve()}"
                    )

                if filepath.stat().st_size == 0:
                    raise RuntimeError(
                        f"Generated PDF is empty: {filepath.resolve()}"
                    )

                logger.info(
                    "Planner PDF generated: %s (%s bytes)",
                    filepath.resolve(),
                    filepath.stat().st_size,
                )

                return {
                    "success": True,
                    "filename": filename,
                    "pdf_filename": pdf_filename,
                    "filepath": str(filepath.resolve()),
                    "file_size": filepath.stat().st_size,
                }

            except Exception as e:
                logger.exception("PDF generation error")

                return {
                    "success": False,
                    "error": f"Failed to generate PDF: {e}",
                }

        self.planner_tools = [
            create_action_plan,
        ]
    def _init_verifier_tools(self):
        """Initialize Verifier Agent tools - Handles human validation only"""
        
        @tool
        def validate_findings(findings: dict, validation_criteria: dict = None):
            """Validate analysis findings with human-in-the-loop process"""
            if validation_criteria is None:
                validation_criteria = {
                    "min_confidence": 0.8,
                    "require_human_review": True,
                    "critical_threshold": 0.95
                }
            
            validation_result = {
                "timestamp": datetime.now().isoformat(),
                "findings_summary": str(findings)[:200] + "...",
                "validation_status": "pending_human_review",
                "confidence_score": 0.85,
                "requires_escalation": validation_criteria.get("require_human_review", True),
                "next_steps": [
                    "Human expert review required",
                    "Validate against historical patterns",
                    "Confirm with operations team"
                ]
            }
            
            return validation_result

        self.verifier_tools = [
            validate_findings
        ]

    def _init_executor_tools(self):
        """Initialize Executor Agent tools - Sends notifications and call MES API to execute validated actions"""
        
        @tool
        def send_email_notification(subject: str, message: str, pdf_filename: str = None, priority: str = "Normal"):
            """Send email notification using SES with optional PDF link"""
            result = self.execute_email_send(subject, message, pdf_filename)
            logger.info(f"Email notification sent: {subject}")
            return result

        self.executor_tools = [
            send_email_notification
        ]

    def _init_agents(self):
        OUTPUT_RULES = """

=== OUTPUT FORMAT RULES (mandatory) ===
- Maximum 600 words total. Be dense, not decorative.
- Use exactly these sections and nothing else:
  1. KEY FINDINGS (max 5 bullet points)
  2. SUPPORTING DATA (max 1 table, max 10 rows)
  3. GAPS / MISSING DATA (what you could not determine and why)
  4. HANDOFF NOTES (max 3 bullets for the next agent)
- No emoji, no ASCII-art charts, no decorative separators.
- Report only numbers that appear in tool results. Never compute
  totals, percentages, correlations, confidence percentages, or
  dollar amounts yourself. If a number was not returned by a tool,
  write "not available in data" instead.
- Express certainty only as HIGH / MEDIUM / LOW with a one-line reason.

- Every KEY FINDING must end with its data source in brackets:
  [source: <exact tool name>, <row count if known>, <date range>].
  The source must be the exact name of a tool called in this
  conversation (e.g. fetch_defect_records). Never cite an analysis,
  method, section, or report name as a source.
  A finding you cannot attribute to a tool result must not be stated.
- In GAPS / MISSING DATA, classify every gap as exactly one of:
  (a) "absent from database" - only when an explicit tool query for it
      returned empty over a window that covers the data;
  (b) "outside analyzed window" - data may exist beyond the days_back
      window used;
  (c) "not exposed by available tools" - when no tool returns that
      kind of data at all.
  Never state (a) when (b) or (c) could explain the absence.
- Each KEY FINDING must be followed by one line starting "WHY: " giving
  a 1-2 sentence causal mechanism in plain manufacturing terms (how this
  cause physically produces this defect). If the mechanism is not
  certain, give the most plausible one and label it "(hypothesis)".
"""

        """Initialize the specialized agents"""
        
        # Monitor Agent - Captures & contextualizes data
        self.monitor_agent = Agent(
            model=self._create_model(),
            hooks=[self._observability_hooks],
            tools=self.monitor_tools + [self.execute_sql_tool],
            system_prompt="""You are the Monitor Agent for a Manufacturing Execution System (MES).

Your primary responsibilities:
1. **Capture Manufacturing Events**: Monitor OEE drops, line stoppages, and downtime events
2. **Fetch Historical Context**: Retrieve historical stoppage patterns, operator logs, and work orders
3. **Contextualize Data**: Provide relevant context including batch reports, maintenance records, and shift information

Key monitoring areas:
- OEE metrics and performance drops
- Production line stoppages and downtime events
- Tool changeover times and batch transitions
- Operator performance and shift patterns
- Work order completion rates and delays

When analyzing events, always:
- Fetch relevant historical patterns for comparison
- Include operator, shift, and product context
- Identify time-based patterns (hour, day, shift)
- Correlate events with maintenance schedules
- Provide comprehensive context for analysis

Focus on capturing complete operational context to enable effective root cause analysis.

DATABASE FACTS: There is no Maintenance, maintenance_log, or CMMS table. Maintenance events are recorded as Reason values (e.g. 'Scheduled Maintenance', 'Cleaning', 'Software Error') inside the Downtimes data, which fetch_downtime_events and fetch_historical_patterns already return. Never query tables not returned by your tools.
""" + OUTPUT_RULES
        )
        
        # Analyzer Agent - Identifies root causes and performs reasoning
        self.analyzer_agent = Agent(
            model=self._create_model(),
            hooks=[self._observability_hooks],
            tools=self.analyzer_tools + [self.execute_sql_tool],
            system_prompt="""You are the Analyzer Agent for a Manufacturing Execution System (MES).

Your primary responsibilities:
1. **Root Cause Analysis**: Identify primary and secondary causes of manufacturing issues
2. **Correlation Analysis**: Find relationships between downtime, operators, shifts, and products
3. **Performance Reasoning**: Analyze excessive batch changeover times vs benchmarks
4. **Pattern Recognition**: Identify systematic issues across machines, products, and processes

Analysis focus areas:
- Correlation between downtime and specific shift/operator/product combinations
- Excessive batch changeover time analysis vs industry benchmarks
- Machine performance patterns and efficiency trends
- Quality defect patterns and their root causes
- Systematic vs random failure analysis

Your reasoning process should:
1. Start with the most impactful issues (highest cost, frequency, or risk)
2. Look for statistical correlations and patterns
3. Consider multiple contributing factors
4. Differentiate between symptoms and root causes
5. Rate certainty as HIGH / MEDIUM / LOW based on evidence strength
6. Recommend data-driven solutions

Base every claim on tool-returned data and provide actionable insights for the planning phase. 
DATABASE FACTS: There is no Maintenance, maintenance_log, quality_defects, or CMMS table. Maintenance events are recorded as Reason values inside the Downtimes data returned by analyze_downtime_correlations. Never query tables not returned by your tools.
""" + OUTPUT_RULES
        )
        
        # Planner Agent - Suggests actionable plans
        self.planner_agent = Agent(
            model=self._create_model(),
            hooks=[self._observability_hooks],
            tools=self.planner_tools,
            system_prompt="""You are the Planner Agent for a Manufacturing Execution System (MES).

Your primary responsibilities:
1. **Action Plan Creation**: Develop prioritized, actionable improvement plans in natural language human readable format
3. **Resource Planning**: Estimate resources, timelines, and costs for improvements
4. **Implementation Strategy**: Provide step-by-step implementation guidance

When creating action plans:
- Prioritize by impact (quality, cost, safety, efficiency)
- Provide clear timelines (immediate, short-term, long-term)
- Specify required resources and responsibilities
- Include success metrics and KPIs
- Consider implementation feasibility and risk

Plan structure should include:
1. **Immediate Actions** (0-30 days): Quick wins and critical fixes
2. **Short-term Actions** (1-3 months): Process improvements and training
3. **Long-term Actions** (3-12 months): Strategic investments and upgrades

Always focus on measurable, actionable recommendations that improve manufacturing performance.""" + OUTPUT_RULES
        )
        
        # Verifier Agent - Handles human validation only
        self.verifier_agent = Agent(
            model=self._create_model(),
            hooks=[self._observability_hooks],
            tools=self.verifier_tools,
            system_prompt="""You are the Verifier Agent for a Manufacturing Execution System (MES).

Your primary responsibilities:
1. **Human-in-the-Loop Validation**: Facilitate human expert review of AI findings
2. **Quality Assurance**: Validate analysis findings against established criteria
3. **Alert Management**: Create validation reports for monitoring dashboards

Validation triggers:
- Critical OEE drops (below 60%)
- Extended downtime events (>2 hours)
- Quality issues with high severity (>3/5)
- Maintenance overdue warnings
- Unusual pattern detection

Validation process:
1. Check findings against historical baselines
2. Assess confidence levels of analysis
3. Determine need for human expert review
4. Escalate critical issues appropriately
5. Track validation outcomes for continuous improvement

Human validation criteria:
- Complex root cause scenarios
- High-impact business decisions
- Safety-related findings
- Strategic investment recommendations
- Unusual or unprecedented patterns

Always maintain audit trails and ensure validation results are properly documented. 
Note: Email notifications are handled by the Executor Agent.""" + OUTPUT_RULES
        )

        # Executor Agent - Sends email notification and call MES APIs to execute actions
        self.executor_agent = Agent(
            model=self._create_model(),
            hooks=[self._observability_hooks],
            tools=self.executor_tools,
            system_prompt="""You are the Executor Agent for a Manufacturing Execution System (MES).

Your primary responsibilities:
1. **Action Plan Execution**: Transform human understandable action plan into MES specific technical action items
2. **Implementation Strategy**: Receive implementation strategy in terms of medium, short term and long term and take action as appropriate 
3. **Email Generation**: Receive comprehensive PDF reports with findings and recommendations and send it in a summarized as well as detailed text through one email
4. **MES API Execution**: Based on actionable plan, execute MES API for immediate actionable item

When executing action plans:
- Accept actionable plan from planner agent
- Draft email to emphasize on four factors (quality, cost, safety, efficiency)
- Send only one email for short-term and long-term action items
- Execute MES API for immediate action item

Email report should include:
1. Detail of all short-term and long-term actionable items
2. Provide detail of which manufacturing department needs to take the action
3. Provide summary of all the issues, findings, root cause analysis
4. Detailed report attached as received from planner agent

When sending email notifications, it is mandatory to pass the report filename provided in your task instructions, include it in the send_email_notification call to attach the PDF link in format(https://dfmw0zqekwl4n.cloudfront.net/proxy/8501/pdf=pdf_filename.pdf?pdf=pdf_filename.pdf) to the email.

Always focus on clear and concise email body with actionable recommendations, ownership, timeline and risks if not done on time.""" + OUTPUT_RULES
        )

    def _init_supervisor_agent(self):
        """Initialize the Supervisor Agent that orchestrates the workflow"""

        SUPERVISOR_OUTPUT_RULES = """

=== OUTPUT RULES (mandatory) ===
- No emoji, no ASCII-art charts, no decorative separators.
- Report only numbers that appear in tool results or subagent reports.
  Never compute totals, percentages, correlations, confidence
  percentages, or dollar amounts yourself. If a number was not
  returned by a tool, write "not available in data" instead.
- Never compare values against industry standards, benchmarks,
  "world-class" figures, or "required" durations unless those values
  appear in tool results.
- Express certainty only as HIGH / MEDIUM / LOW with a one-line reason.
- Your final report is a detailed synthesis for a human domain expert.
  For each finding include: what was observed, the causal mechanism
  (WHY), the supporting evidence with its tool source, and what remains
  uncertain. Do not compress away detail from subagent reports; the
  word limits that apply to subagents do NOT apply to you.
- Preserve exact numbers from subagent reports verbatim. Never
  recompute, re-split, or restate counts (such as per-machine splits);
  copy them as given, with their sources.
- Structure the final report with exactly these numbered sections, in
  this order, each section heading immediately followed by a line
  "Source: <the tools/agents the section draws on>":
  1. Defect Occurrence Summary (include a per-occurrence comparison table)
  2. Maintenance Correlation Findings
  3. Root Cause Hypotheses (ranked; each with WHY mechanism and
     HIGH/MEDIUM/LOW certainty)
  4. Data Reliability Flags (inconsistencies or anomalies in the data
     itself that must be resolved before conclusions can be trusted)
  5. Gaps / Missing Data (classified per the GAPS rule)
  6. Action Plan (immediate / short-term / long-term, from the Planner)
  7. Verification Outcome and Conditions (from the Verifier)
  8. Notification Status (from the Executor)
"""

        
        @tool
        def call_monitor_agent(prompt: str):
            """Call the Monitor Agent to capture operational data"""
            return self._call_agent_with_retry("monitor", self.monitor_agent, prompt)
        
        @tool
        def call_analyzer_agent(prompt: str):
            """Call the Analyzer Agent to perform root cause analysis"""
            return self._call_agent_with_retry("analyzer", self.analyzer_agent, prompt)
        
        @tool
        def call_planner_agent(prompt: str):
            """Call the Planner Agent to create action plans"""
            return self._call_agent_with_retry("planner", self.planner_agent, prompt)
        
        @tool
        def call_verifier_agent(prompt: str):
            """Call the Verifier Agent to validate findings"""
            return self._call_agent_with_retry("verifier", self.verifier_agent, prompt)
        
        @tool
        def call_executor_agent(prompt: str):
            """"Call the Executor Agent to execute plans and send notifications"""
            return self._call_agent_with_retry("executor", self.executor_agent, prompt)
        
        self.supervisor_agent = Agent(
            model=self._create_model(),
            hooks=[self._observability_hooks],
            tools=[call_monitor_agent, call_analyzer_agent, call_planner_agent, call_verifier_agent, call_executor_agent],
            system_prompt="""You are the Supervisor Agent for the Manufacturing Execution System (MES) AI workflow.

Your primary responsibility is to orchestrate the complete defect analysis workflow by coordinating five specialized agents:

1. **Monitor Agent**: Captures operational data and contextualizes manufacturing events
2. **Analyzer Agent**: Performs root cause analysis and identifies correlations
3. **Planner Agent**: Creates actionable improvement plans
4. **Verifier Agent**: Validates findings and manages human validation
5. **Executor Agent**: Executes action plans and sends email notifications

**Workflow Process:**
1. Receive defect analysis request with defect type, time period, and analysis scope parameters
2. Call Monitor Agent to capture comprehensive operational data based on enabled scope
3. Call Analyzer Agent to perform root cause analysis using monitoring data and scope
4. Call Planner Agent to create action plans based on analysis results and scope
5. Call Verifier Agent to validate findings within scope
6. Call Executor Agent to execute immediate actions and send email notifications, passing the final report filename given in your task instructions
7. Compile complete analysis results with all agent outputs

**Analysis Scope Parameters:**
- include_oee: Enable/disable OEE performance analysis
- include_downtime: Enable/disable downtime and stoppages analysis
- include_changeover: Enable/disable batch changeover analysis
- include_maintenance: Enable/disable maintenance correlation analysis

**Key Responsibilities:**
- Ensure proper data flow between agents with scope considerations
- Maintain analysis context throughout the workflow
- Coordinate timing and sequencing of agent activities
- Compile comprehensive results from all agents
- Handle error recovery and workflow continuity
- Provide executive summary of complete analysis
- Respect analysis scope limitations and focus areas
- Use Executor Agent for one email notification and action execution

**Critical Workflow Note:**
Your task instructions include the filename under which your final report
will be published. Pass that exact filename to the Executor Agent for the
email notification link. Do not invent or alter it.

**Output Format:**
Always return a structured analysis result containing:
- Defect type and analysis parameters including scope settings
- Monitoring results with operational context within enabled scope
- Root cause analysis with HIGH/MEDIUM/LOW certainty ratings for enabled areas
- Action plans with timelines and resources for enabled scope
- Verification results with validation status
- Execution results with notification status
- Executive summary with key findings and recommendations

Focus on ensuring each agent receives appropriate context and scope parameters, and that the complete workflow produces actionable, validated insights for manufacturing quality improvement within the specified analysis scope. All email notifications should be handled through the Executor Agent.""" + SUPERVISOR_OUTPUT_RULES
        )

    def _save_agent_metrics(self, agent_name: str, agent_obj):
        """Write the agent's token/cycle metrics into the current run folder."""
        try:
            run_dir = getattr(self, "current_run_dir", None)
            metrics = getattr(agent_obj, "event_loop_metrics", None)
            if run_dir is None or metrics is None:
                return
            summary = metrics.get_summary() if hasattr(metrics, "get_summary") else str(metrics)
            ts = datetime.now().strftime('%H%M%S')
            (run_dir / f"{ts}_{agent_name}_metrics.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Metrics capture failed for {agent_name}: {e}")

    def _save_agent_transcript(self, agent_name: str, agent_obj):
        """Dump the agent's full internal conversation: every turn,
        every tool call with arguments, every raw tool result."""
        try:
            run_dir = getattr(self, "current_run_dir", None)
            if run_dir is None:
                return
            messages = getattr(agent_obj, "messages", None)
            if messages is None:
                return
            ts = datetime.now().strftime('%H%M%S')
            (run_dir / f"{ts}_{agent_name}_transcript.json").write_text(
                json.dumps(messages, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Transcript capture failed for {agent_name}: {e}")

    def _call_agent_with_retry(self, agent_name: str, agent_obj, prompt: str):
        """Run an agent turn; on failure (timeout/connection), retry once."""
        last_error = None
        # Emitted before _active_agent switches, so this event is tagged
        # with the delegator (supervisor) — prompts flow downward.
        self._emit("agent_started", {
            "agent_name": agent_name,
            "delegation_prompt": str(prompt),
        })
        prev_agent = self._active_agent
        self._active_agent = agent_name
        started = time.time()
        try:
            for attempt in (1, 2):
                try:
                    result = agent_obj(prompt)
                    self._save_agent_output(agent_name, prompt, result)
                    self._save_agent_transcript(agent_name, agent_obj)
                    self._save_agent_metrics(agent_name, agent_obj)
                    self._emit("agent_completed", {
                        "agent_name": agent_name,
                        "duration_s": round(time.time() - started, 1),
                        "result_preview": self._preview(result, 500),
                        "metrics": self._metrics_snapshot(agent_obj),
                    })
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{agent_name} agent attempt {attempt} failed: {e}"
                        + (" - retrying once" if attempt == 1 else " - giving up"))
                    self._emit("agent_retry" if attempt == 1 else "agent_failed", {
                        "agent_name": agent_name,
                        "attempt": attempt,
                        "error": str(e),
                    })
            return (f"[{agent_name} agent unavailable after 2 attempts: {last_error}. "
                    f"Proceed with available information and state this gap explicitly.]")
        finally:
            self._active_agent = prev_agent

    def _save_agent_output(self, agent_name: str, prompt: str, result):
        """Write an agent's input and output into the current run folder."""
        try:
            run_dir = getattr(self, "current_run_dir", None)
            if run_dir is None:
                return
            ts = datetime.now().strftime('%H%M%S')
            (run_dir / f"{ts}_{agent_name}_prompt.txt").write_text(str(prompt), encoding="utf-8")
            (run_dir / f"{ts}_{agent_name}_output.txt").write_text(str(result), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Run capture failed for {agent_name}: {e}")

    def run_defect_analysis(self, defect_type: str, days_back: int = 7, include_oee: bool = True, 
                           include_downtime: bool = True, include_changeover: bool = True, 
                           include_maintenance: bool = True):
        """Run comprehensive defect analysis using supervisor agent orchestration"""
        
        scope_summary = []
        if include_oee:
            scope_summary.append("OEE Analysis")
        if include_downtime:
            scope_summary.append("Downtime Analysis")
        if include_changeover:
            scope_summary.append("Changeover Analysis")
        if include_maintenance:
            scope_summary.append("Maintenance Correlation")
        
        scope_text = ", ".join(scope_summary) if scope_summary else "Basic Analysis"
        
        start_time = datetime.now()

        # --- run artifact capture ---
        run_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{re.sub(r'[^A-Za-z0-9]+', '', defect_type)}"
        self.current_run_dir = Path("runs") / run_id
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        (self.current_run_dir / "params.json").write_text(json.dumps({
            "defect_type": defect_type,
            "days_back": days_back,
            "include_oee": include_oee,
            "include_downtime": include_downtime,
            "include_changeover": include_changeover,
            "include_maintenance": include_maintenance,
            "model_id": os.getenv("MES_MODEL_ID", "unknown"),
            "started": start_time.isoformat(),
        }, indent=2), encoding="utf-8")
        run_log_handler = logging.FileHandler(self.current_run_dir / "run.log", encoding="utf-8")
        run_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(run_log_handler)

        # Fresh event stream per run (the manager may be reused).
        self.event_log = []
        self._event_seq = itertools.count(1)
        self._emit("run_started", {
            "defect_type": defect_type,
            "days_back": days_back,
            "scope": scope_text,
            "run_id": run_id,
        })

        try:
            # Create comprehensive prompt for supervisor agent
            final_report_filename = f"MES_Final_Report_{start_time.strftime('%Y%m%d_%H%M%S')}"

            supervisor_prompt = f"""
            Execute comprehensive defect analysis workflow for defect type '{defect_type}' over the last {days_back} days.
            
            Analysis Scope Configuration:
            - OEE Analysis: {'Enabled' if include_oee else 'Disabled'}
            - Downtime Analysis: {'Enabled' if include_downtime else 'Disabled'}
            - Changeover Analysis: {'Enabled' if include_changeover else 'Disabled'}
            - Maintenance Correlation: {'Enabled' if include_maintenance else 'Disabled'}
            
            Execute the following workflow steps:
            
            1. **Monitor Phase**: Call Monitor Agent to capture operational data
               - Focus on {defect_type} defect occurrences and context
               - Include enabled analysis areas: {scope_text}
               - Gather historical patterns and operational context
            
            2. **Analysis Phase**: Call Analyzer Agent for root cause analysis
               - Analyze monitoring data for {defect_type} root causes
               - Focus on enabled correlation areas: {scope_text}
               - Provide statistical confidence and impact assessment
            
            3. **Planning Phase**: Call Planner Agent to create action plans
               - Develop comprehensive improvement plans for {defect_type}
               - Address enabled improvement areas: {scope_text}
               - Include immediate, short-term, and long-term actions
            
            4. **Verification Phase**: Call Verifier Agent to validate findings
               - Validate analysis results and action plans
               - Determine notification requirements
               - Assess need for human expert review
            
            5. **Execution Phase**: Call Executor Agent to execute actions
               - Execute immediate action items
               - Send email notifications with detailed reports
               - Include the final report filename {final_report_filename}.pdf in the email link
               - Coordinate with manufacturing departments
            
            Ensure each agent receives appropriate context from previous phases and respects the analysis scope limitations.
            
            IMPORTANT: The final comprehensive report will be published as {final_report_filename}.pdf. Pass exactly this filename to the Executor Agent for the email notification link.
            
            Compile comprehensive results including all agent outputs and provide executive summary.
            """ 
            
                        # Call Supervisor Agent to orchestrate the workflow.
            supervisor_response = self.supervisor_agent(
                supervisor_prompt
            )

            self._save_agent_output(
                "supervisor_final",
                supervisor_prompt,
                supervisor_response,
            )

            self._save_agent_transcript(
                "supervisor_final",
                self.supervisor_agent,
            )

            self._save_agent_metrics(
                "supervisor_final",
                self.supervisor_agent,
            )

                        # Get the Supervisor's complete final report.
            supervisor_results = str(supervisor_response).strip()

            if not supervisor_results:
                raise RuntimeError(
                    "The Supervisor returned an empty final report."
                )

            logger.info(
                "Supervisor final report received: %s characters",
                len(supervisor_results),
            )

            # Convert the Supervisor's complete report into one PDF.
            final_pdf_path = render_markdown_report_pdf(
                markdown_text=supervisor_results,
                filename=final_report_filename,
            )

            final_pdf = Path(final_pdf_path)

            if not final_pdf.exists():
                raise RuntimeError(
                    f"The Supervisor PDF was not created: "
                    f"{final_pdf.resolve()}"
                )

            if final_pdf.stat().st_size == 0:
                raise RuntimeError(
                    f"The Supervisor PDF is empty: "
                    f"{final_pdf.resolve()}"
                )

            logger.info(
                "Supervisor report converted to PDF: %s (%s bytes)",
                final_pdf.resolve(),
                final_pdf.stat().st_size,
            )

            end_time = datetime.now()

            if not supervisor_results or not supervisor_results.strip():
                raise RuntimeError("Supervisor returned an empty final report.")

            final_pdf_path = render_markdown_report_pdf(
                supervisor_results,
                filename=final_report_filename
            )

            if not final_pdf_path:
                raise RuntimeError("PDF generation returned no filepath.")

            final_pdf = Path(final_pdf_path)

            if not final_pdf.exists():
                raise RuntimeError(
                    f"PDF generation returned a path, but the file was not created: "
                    f"{final_pdf.resolve()}"
                )

            if final_pdf.stat().st_size == 0:
                raise RuntimeError(
                    f"The generated PDF is empty: {final_pdf.resolve()}"
                )

            logger.info(
                "Final PDF generated successfully: %s (%s bytes)",
                final_pdf.resolve(),
                final_pdf.stat().st_size,
            )

            end_time = datetime.now()
            
                        
                        # Compile comprehensive results.
            analysis_results = {
                "defect_type": defect_type,
                "analysis_period": days_back,
                "analysis_scope": {
                    "include_oee": include_oee,
                    "include_downtime": include_downtime,
                    "include_changeover": include_changeover,
                    "include_maintenance": include_maintenance,
                    "scope_summary": scope_text,
                },
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": (
                    end_time - start_time
                ).total_seconds(),
                "supervisor_orchestration": supervisor_results,

                # Confirmed PDF information.
                "pdf_generated": True,
                "pdf_path": str(final_pdf.resolve()),
                "pdf_filename": final_pdf.name,
                "pdf_size": final_pdf.stat().st_size,

                "workflow_status": "completed",
                "executive_summary": f"""
Comprehensive defect analysis completed for {defect_type} defects
over {days_back} days using Supervisor Agent orchestration.

Analysis scope: {scope_text}

The Supervisor Agent coordinated the specialized agents to:
- Monitor operational data and manufacturing events
- Analyze root causes and correlations
- Plan actionable improvement strategies
- Verify findings and recommendations
- Execute immediate actions and send notifications

Total analysis duration:
{(end_time - start_time).total_seconds():.2f} seconds

PDF report:
{final_pdf.name}
                """.strip(),
                "status": "completed",
            }

            self._emit("run_completed", {
                "status": "completed",
                "duration_s": round((end_time - start_time).total_seconds(), 1),
                "pdf_filename": final_pdf.name,
            })
            return analysis_results

        except Exception as e:
            logger.exception(
    "Error in supervisor-orchestrated defect analysis workflow"
)
            self._emit("run_failed", {"error": str(e)})
            return {
                'defect_type': defect_type,
                'analysis_period': days_back,
                'analysis_scope': {
                    'include_oee': include_oee,
                    'include_downtime': include_downtime,
                    'include_changeover': include_changeover,
                    'include_maintenance': include_maintenance,
                    'scope_summary': scope_text
                },
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
        
        finally:
            self._save_event_log()
            logging.getLogger().removeHandler(run_log_handler)
            run_log_handler.close()

    def get_monitor_agent(self):
        """Get the monitor agent"""
        return self.monitor_agent
    
    def get_analyzer_agent(self):
        """Get the analyzer agent"""
        return self.analyzer_agent
    
    def get_planner_agent(self):
        """Get the planner agent"""
        return self.planner_agent
    
    def get_executor_agent(self):
        """Get the Executor agent"""
        return self.executor_agent
    
    def get_verifier_agent(self):
        """Get the verifier agent"""
        return self.verifier_agent
    
    def get_supervisor_agent(self):
        """Get the supervisor agent"""
        return self.supervisor_agent
    
    def get_defect_types(self, days_back):
        """Execute SQL query directly without going through agent"""
        # Validate input
        days_back = int(days_back)
        if days_back < 0 or days_back > 3650:
            raise ValueError("days_back must be between 0 and 3650")
        
        # Calculate the cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        sql_query = """
        SELECT DISTINCT d.DefectType
        FROM Defects d
        JOIN QualityControl qc ON d.CheckID = qc.CheckID
        WHERE date(qc.Date) >= ?
        ORDER BY d.DefectType
        """
        
        return self._execute_safe_query(sql_query, (cutoff_date,))

    def get_defect_preview(self, defect_type):
        """Execute SQL query directly without going through agent"""
        # Calculate the cutoff date (30 days back)
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        sql_query = """
        SELECT 
            COUNT(*) as TotalOccurrences,
            AVG(d.Severity) as AvgSeverity,
            COUNT(DISTINCT wo.MachineID) as MachinesAffected,
            COUNT(DISTINCT wo.ProductID) as ProductsAffected,
            COUNT(DISTINCT d.RootCause) as RootCauseVariety,
            MAX(qc.Date) as LastOccurrence
        FROM 
            Defects d
        JOIN 
            QualityControl qc ON d.CheckID = qc.CheckID
        JOIN 
            WorkOrders wo ON qc.OrderID = wo.OrderID
        WHERE 
            d.DefectType = ?
            AND date(qc.Date) >= ?
        """
        
        return self._execute_safe_query(sql_query, (defect_type, cutoff_date))
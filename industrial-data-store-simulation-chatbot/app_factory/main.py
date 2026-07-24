"""
Main entry point for the Manufacturing Operations Hub.

Business users land directly on the Daily Production Meeting dashboard (answers
first). A persistent sidebar switch moves between the dashboard and the MES
Insight Chat, so there is no separate "pick an app" screen to get through.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the app modules
from mes_chat.chat_interface import run_mes_chat
from production_meeting.dashboard import run_production_meeting

# Page configuration
st.set_page_config(
    page_title="Manufacturing Operations Hub",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# The two workspaces available in the hub.
DASHBOARD = "production_meeting"
CHAT = "mes_chat"

VIEW_LABELS = {
    DASHBOARD: "Daily Production Meeting",
    CHAT: "MES Insight Chat",
}


def render_workspace_switch():
    """Persistent sidebar control for moving between the two workspaces."""
    with st.sidebar:
        st.markdown("### 🏭 Operations Hub")
        st.caption("E-bike Manufacturing Facility")

        choice = st.radio(
            "Workspace",
            options=[DASHBOARD, CHAT],
            format_func=lambda v: VIEW_LABELS[v],
            key="app_mode",
            label_visibility="collapsed",
        )
        st.divider()
        return choice


def main():
    """Main application entry point."""

    # Land on the dashboard by default so the first screen shows today's answers,
    # not a menu.
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = DASHBOARD

    # Sidebar workspace switch is always available; each workspace appends its own
    # settings below it.
    selected = render_workspace_switch()

    if selected == CHAT:
        run_mes_chat()
    else:
        run_production_meeting()


if __name__ == "__main__":
    main()

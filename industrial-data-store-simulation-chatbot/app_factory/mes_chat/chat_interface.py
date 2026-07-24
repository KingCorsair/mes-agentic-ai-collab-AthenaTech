"""
Agentic MES chatbot application using Strands SDK.

This application provides an intelligent agentic chat interface with multi-step analysis capabilities through Strands agents.
"""

import json
import logging
import os
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app_factory.shared.database import DatabaseManager
# Removed bedrock_utils dependency - using simplified model management
from mes_agents.agent_manager import MESAgentManager
from mes_agents.config import AgentConfig

# Configuration
load_dotenv()
proj_dir = os.path.abspath('')
db_path = os.path.join(proj_dir, 'mes.db')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Developer/admin controls (model choice, timings, agent internals) are hidden from
# business users by default. Set MES_SHOW_ADVANCED=1 to reveal them.
SHOW_ADVANCED = os.getenv("MES_SHOW_ADVANCED", "").strip().lower() in ("1", "true", "yes")

# Initialize database tool for fallback
db_tool = DatabaseManager(db_path)

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')



def reset_chat():
    """Reset the agent chat state"""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your manufacturing analyst. Ask me anything about production, quality, equipment, or inventory — for example, *\"What was our completion rate yesterday?\"*"
        }
    ]
    st.session_state.conversation_history = []
    st.session_state.last_result = None
    st.session_state.progress = []
    
    # Reset the persistent agent conversation history
    if 'agent_manager' in st.session_state:
        st.session_state.agent_manager.reset_conversation()


def display_progress_updates(progress_updates: List[Dict[str, Any]]):
    """
    Display agent progress updates in the UI.
    
    Args:
        progress_updates: List of progress update dictionaries
    """
    if not progress_updates:
        return
    
    with st.expander("🔄 Agent Analysis Progress", expanded=True):
        for i, update in enumerate(progress_updates):
            status_icon = {
                'initializing': '🚀',
                'planning': '🧠', 
                'executing': '⚙️',
                'analyzing': '📊',
                'completing': '✅',
                'completed': '🎉',
                'error': '❌'
            }.get(update.get('status', 'executing'), '⚙️')
            
            timestamp = update.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = ''
            else:
                time_str = ''
            
            st.write(f"{status_icon} **Step {update.get('step', i+1)}**: {update.get('message', 'Processing...')} {time_str}")


def display_agent_response(response: Dict[str, Any], message_index: int):
    """
    Display agent response with enhanced formatting.
    
    Args:
        response: Agent response dictionary
        message_index: Index of the message for unique keys
    """
    if not response.get('success', True):
        # Display error response
        st.error(f"Analysis Error: {response.get('error', 'Unknown error')}")
        
        if response.get('suggested_actions'):
            st.markdown("**Suggested Actions:**")
            for action in response['suggested_actions']:
                st.write(f"• {action}")
        
        if response.get('recovery_options'):
            st.markdown("**Recovery Options:**")
            for option in response['recovery_options']:
                st.write(f"• {option}")
        
        return
    
    # Display successful analysis
    analysis_content = response.get('analysis', '')
    if analysis_content:
        st.markdown(analysis_content)
    
    # Display progress updates if available (developer/admin view only)
    progress_updates = response.get('progress_updates', [])
    if progress_updates and SHOW_ADVANCED:
        display_progress_updates(progress_updates)

    # Execution telemetry (timing, agent type, tools used) is developer-facing and
    # only shown when the advanced/admin flag is set.
    if SHOW_ADVANCED:
        execution_time = response.get('execution_time', 0)
        analysis_depth = response.get('analysis_depth', 'standard')
        stats_text = f"**Analysis Time:** {execution_time:.2f}s | **Analysis Depth:** {analysis_depth.title()} | **Agent Type:** Strands Agent"
        st.caption(stats_text)

        capabilities_used = response.get('capabilities_used', [])
        if capabilities_used:
            formatted_tools = []
            for tool in capabilities_used:
                if tool == 'mes_analysis_tool':
                    formatted_tools.append('MES Analysis')
                elif tool == 'run_sqlite_query':
                    formatted_tools.append('SQL Query')
                elif tool == 'get_database_schema':
                    formatted_tools.append('Schema Analysis')
                elif tool == 'create_intelligent_visualization':
                    formatted_tools.append('Visualization')
                else:
                    formatted_tools.append(tool.replace('_', ' ').title())
            st.caption(f"**Tools Used:** {' • '.join(formatted_tools)}")

    # Display follow-up suggestions
    follow_ups = response.get('follow_up_suggestions', [])
    if follow_ups:
        st.markdown("**💡 Suggested Follow-up Analyses:**")
        cols = st.columns(min(len(follow_ups), 2))
        for i, suggestion in enumerate(follow_ups):
            with cols[i % 2]:
                if st.button(suggestion, key=f"followup_{message_index}_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.session_state["process_query"] = suggestion
                    st.rerun()


def display_agent_status_sidebar(agent_manager: MESAgentManager):
    """
    Display agent status and configuration in the sidebar.
    
    Args:
        agent_manager: The MES agent manager instance
    """
    st.subheader("🤖 Agent Status")
    
    status = agent_manager.get_agent_status()
    
    # Agent status indicator
    status_color = {
        'ready': '🟢',
        'not_available': '🔴', 
        'error': '🟡'
    }.get(status.get('status'), '🔴')
    
    st.write(f"{status_color} **Status**: {status.get('status', 'unknown').title()}")
    
    if status.get('status') == 'ready':
        st.success("Agent ready for analysis")
        
        # Display agent capabilities
        capabilities = status.get('capabilities', [])
        if capabilities:
            st.markdown("**Capabilities:**")
            for cap in capabilities:
                st.write(f"• {cap.title()}")
    
    elif status.get('status') == 'not_available':
        st.error("Agent not available")
        if status.get('error'):
            st.write(f"Error: {status['error']}")
    
    # Display configuration
    config = status.get('config', {})
    if config:
        with st.expander("⚙️ Agent Configuration"):
            st.write(f"**Model**: {config.get('model', 'Unknown')}")
            st.write(f"**Analysis Depth**: {config.get('analysis_depth', 'standard').title()}")
            st.write(f"**Timeout**: {config.get('timeout', 120)}s")
            st.write(f"**Max Steps**: {config.get('max_query_steps', 5)}")
            st.write(f"**Progress Updates**: {'✅' if config.get('progress_updates_enabled') else '❌'}")
    
    # Display integration info
    integration_info = agent_manager.get_integration_info()
    with st.expander("🔗 Integration Details"):
        st.write(f"**Framework**: {integration_info.get('integration_type', 'Unknown')}")
        st.write(f"**Agent Ready**: {'✅' if integration_info.get('agent_ready') else '❌'}")
        st.write(f"**Database**: {integration_info.get('database_backend', 'Unknown')}")
        st.write(f"**Visualization**: {integration_info.get('visualization_library', 'Unknown')}")


async def process_agent_query(agent_manager: MESAgentManager, query: str) -> Dict[str, Any]:
    """
    Process query using the agent manager.
    
    Args:
        agent_manager: The MES agent manager
        query: User query string
        
    Returns:
        Agent response dictionary
    """
    # Prepare context from conversation history
    context = {
        'history': st.session_state.get('conversation_history', []),
        'previous_results': [st.session_state.get('last_result')] if st.session_state.get('last_result') else [],
        'preferences': {
            'analysis_depth': st.session_state.get('analysis_depth', 'standard'),
            'include_visualizations': True,
            'include_follow_ups': True
        }
    }
    
    # Process the query
    result = await agent_manager.process_query(query, context)
    
    # Update conversation history
    st.session_state.conversation_history.append({
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'summary': result.get('analysis', '')[:200] + '...' if result.get('analysis') else 'Analysis completed'
    })
    
    # Store last result
    st.session_state.last_result = result
    
    return result

def run_mes_chat():
    """Main function to run the agent-enabled MES chat interface"""
    
    # Page configuration
    st.header("MES Insight Chat")
    st.caption("Ask questions about your factory in plain English and get answers from live MES data.")
    
    # Initialize session state for agent chat
    if "messages" not in st.session_state:
        reset_chat()
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
        
    if "progress" not in st.session_state:
        st.session_state.progress = []
    
    # Initialize agent manager with default configuration (store in session state for persistence)
    if 'agent_manager' not in st.session_state:
        try:
            # Create agent configuration with defaults
            agent_config = AgentConfig()
            
            # Initialize agent manager and store in session state
            st.session_state.agent_manager = MESAgentManager(agent_config)
            
        except Exception as e:
            st.error(f"Failed to initialize agent manager: {e}")
            st.stop()
    
    agent_manager = st.session_state.agent_manager
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("MES Insight Chat")

        # Primary, business-user action: start a fresh conversation.
        st.button("New conversation", on_click=reset_chat, use_container_width=True)

        # What this assistant can do (plain language, no internals).
        with st.expander("What can I ask?"):
            st.markdown("""
            Ask about your factory in plain English — for example:

            - *"What was our completion rate yesterday?"*
            - *"Which products have the highest defect rates?"*
            - *"What machines need maintenance this week?"*
            - *"Which materials are below reorder level?"*

            The assistant reads live MES data, does the analysis, and shows
            charts where they help.
            """)

        # Developer/admin controls: model choice, analysis depth, agent status.
        # Hidden from business users unless MES_SHOW_ADVANCED is set.
        if SHOW_ADVANCED:
            st.divider()
            with st.expander("⚙️ Advanced (admin)"):
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    options=["quick", "standard", "comprehensive"],
                    index=1,
                    help="Choose how deep the agent should analyze queries",
                    key="analysis_depth"
                )
                if agent_manager.config.analysis_depth != analysis_depth:
                    agent_manager.config.analysis_depth = analysis_depth
                    agent_manager.update_config(agent_manager.config)

                available_models = agent_manager.config.SUPPORTED_MODELS
                model_display_names = agent_manager.config.get_model_display_names()
                current_model_index = 0
                if agent_manager.config.default_model in available_models:
                    current_model_index = available_models.index(agent_manager.config.default_model)

                selected_model_index = st.selectbox(
                    "AI Model",
                    range(len(available_models)),
                    index=current_model_index,
                    format_func=lambda x: model_display_names.get(available_models[x], available_models[x]),
                    help="Select the AI model for analysis",
                    key="selected_model"
                )
                selected_model_id = available_models[selected_model_index]
                if agent_manager.config.default_model != selected_model_id:
                    agent_manager.config.default_model = selected_model_id
                    agent_manager.update_config(agent_manager.config)

            display_agent_status_sidebar(agent_manager)
    
    # Initialize process_query if it doesn't exist
    if "process_query" not in st.session_state:
        st.session_state["process_query"] = None

    # Main panel with chat interface
    main_col = st.container()

    with main_col:
        # Conversation is front and center: display the chat history first.
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    if isinstance(message["content"], dict):
                        display_agent_response(message["content"], i)
                    else:
                        st.markdown(message["content"])

        # Example questions are onboarding scaffolding: show them only while the
        # conversation is empty (just the welcome message), as a few clickable chips.
        conversation_empty = len(st.session_state.messages) <= 1
        if conversation_empty:
            try:
                questions_path = Path(__file__).parent.parent / 'data' / 'sample_questions.json'
                if not questions_path.exists():
                    questions_path = Path('sample_questions.json')
                with open(questions_path, 'r', encoding="utf-8") as file:
                    question_data = json.load(file)
                    category_questions = question_data['categories']
            except Exception as e:
                logger.warning(f"Could not load example questions: {e}")
                category_questions = {}

            if category_questions:
                st.caption("Try one of these to get started:")
                # Flatten a couple of examples per category into one row of chips.
                starters = []
                for cat_key in ["🏭 Production", "⚠️ Quality", "🔧 Machines", "📦 Inventory"]:
                    for q in category_questions.get(cat_key, [])[:1]:
                        starters.append(q)

                cols = st.columns(2)
                for idx, q in enumerate(starters):
                    with cols[idx % 2]:
                        if st.button(q, key=f"starter_{hash(q)}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": q})
                            st.session_state["process_query"] = q
                            st.rerun()

        # Chat input (Streamlit pins this to the bottom of the page).
        user_input = st.chat_input("Ask about production, quality, equipment, or inventory…")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state["process_query"] = user_input
            st.rerun()
    
    # Process agent query if needed
    if st.session_state["process_query"]:
        query = st.session_state["process_query"]
        st.session_state["process_query"] = None  # Clear the flag
        
        # Check if agent is ready
        if not agent_manager.is_ready():
            st.error("The assistant isn't available right now. Please refresh the page in a moment, or contact your administrator if this continues.")
            return

        # Create a placeholder for progress updates
        progress_placeholder = st.empty()

        with st.spinner("Analyzing your question…"):
            try:
                # Process the query asynchronously
                response = asyncio.run(process_agent_query(agent_manager, query))
                
                # Add the response to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Clear progress placeholder
                progress_placeholder.empty()
                
                # Rerun to display the new message
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error processing agent query: {e}")
                error_response = {
                    'success': False,
                    'error': "Something went wrong while analyzing that question.",
                    'message': 'Failed to process query',
                    'query': query,
                    'suggested_actions': [
                        'Try rephrasing your question, or ask a simpler version',
                        'Ask about a shorter time period',
                        'If this keeps happening, contact your administrator'
                    ]
                }
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_response
                })
                
                progress_placeholder.empty()
                st.rerun()

# This allows the module to be run directly for testing
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="MES Insight Chat", 
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    run_mes_chat()
import streamlit as st
import asyncio
import threading
import os
from dotenv import load_dotenv

from message_bus import NATSMessageBus
from pydantic_agent import create_pydantic_agent

load_dotenv()

st.set_page_config(page_title="Enhanced ROSA Agent", layout="wide", initial_sidebar_state="expanded")
st.title("Enhanced ROSA Agent - Pydantic AI")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Background Event Loop Setup
if "loop_thread" not in st.session_state:
    loop = asyncio.new_event_loop()
    
    def start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    
    thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    thread.start()
    
    st.session_state.loop_thread = thread
    st.session_state.event_loop = loop

# Agent Initialization
if "agent" not in st.session_state:
    with st.spinner("Connecting to Message Bus..."):
        async def init_agent():
            nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
            bus = NATSMessageBus(nats_url)
            await bus.connect()
            return await create_pydantic_agent(bus)
        
        future = asyncio.run_coroutine_threadsafe(
            init_agent(), 
            st.session_state.event_loop
        )
        try:
            st.session_state.agent = future.result(timeout=10)
            st.success("Agent Initialized & Connected!")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# Sidebar for example queries
with st.sidebar:
    st.markdown("### Example Queries")
    st.markdown("""
    **Navigation:**
    - "Navigate to x=2.0, y=1.5"
    - "Go in front of the table with a banana"
    - "What is my current position?"
    - "What semantic locations are available?"
    
    **ROS Inspection:**
    - "List all topics"
    - "List all nodes"
    - "What is the info for /cmd_vel topic?"
    - "Who publishes to /camera/raw_image?"
    - "Who subscribes to /scan?"
    - "Echo /odom topic"
    - "Subscribe to /cmd_vel"
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask ROSA (e.g., 'Navigate to x=2.0, y=1.5' or 'List all topics')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.get("agent"):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        st.session_state.agent.run(prompt),
                        st.session_state.event_loop
                    )
                    response = future.result(timeout=60)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("Agent is not initialized.")


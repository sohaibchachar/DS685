import streamlit as st
import asyncio
import threading
import os
from dotenv import load_dotenv

from message_bus import NATSMessageBus
from agent import create_agent, set_global_loop

load_dotenv()

st.set_page_config(page_title="ROSA Agent", layout="wide", initial_sidebar_state="expanded")
st.title("ROSA Agent - Decoupled Architecture")

with st.sidebar:
    st.markdown("### Example Queries")
    st.markdown("""
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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loop_thread" not in st.session_state:
    loop = asyncio.new_event_loop()
    
    def start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    
    thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    thread.start()
    
    st.session_state.loop_thread = thread
    st.session_state.event_loop = loop
    
    set_global_loop(loop)

# Agent Initialization
if "agent" not in st.session_state:
    with st.spinner("Connecting to Message Bus..."):
        async def init_agent():
            nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
            bus = NATSMessageBus(nats_url)
            await bus.connect()
            return create_agent(bus)
        
        future = asyncio.run_coroutine_threadsafe(
            init_agent(), 
            st.session_state.event_loop
        )
        try:
            st.session_state.agent = future.result(timeout=10)
            st.success("Agent Initialized & Connected!")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask ROSA (e.g., 'List active nodes')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.get("agent"):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.process(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("Agent is not initialized.")

import streamlit as st
import asyncio
import threading
import os
from dotenv import load_dotenv

from core import MessageBus
from agent import create_agent, set_global_loop

load_dotenv()

st.set_page_config(page_title="ROSA Agent", layout="wide")
st.title("ROSA Agent - Decoupled Architecture")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Background Event Loop Setup ---
# We need a dedicated event loop thread for NATS because Streamlit 
# re-runs the script on every interaction.
if "loop_thread" not in st.session_state:
    loop = asyncio.new_event_loop()
    
    def start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    
    thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    thread.start()
    
    st.session_state.loop_thread = thread
    st.session_state.event_loop = loop
    
    # Pass this loop to the agent module so tools can use it
    set_global_loop(loop)

# --- Agent Initialization ---
if "agent" not in st.session_state:
    with st.spinner("Connecting to Message Bus..."):
        async def init_agent():
            nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
            bus = MessageBus(nats_url)
            await bus.connect()
            return create_agent(bus)
        
        # Run initialization on the background loop
        future = asyncio.run_coroutine_threadsafe(
            init_agent(), 
            st.session_state.event_loop
        )
        try:
            st.session_state.agent = future.result(timeout=10)
            st.success("Agent Initialized & Connected!")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# --- Chat Interface ---

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle Input
if prompt := st.chat_input("Ask ROSA (e.g., 'List active nodes')"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Agent Response
    if st.session_state.get("agent"):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # The agent executor is synchronous, but it calls tools 
                    # that bridge to the async loop via 'async_wrap'
                    response = st.session_state.agent.invoke(
                        {"input": prompt, "chat_history": ""}
                    )
                    output_text = response['output']
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("Agent is not initialized.")
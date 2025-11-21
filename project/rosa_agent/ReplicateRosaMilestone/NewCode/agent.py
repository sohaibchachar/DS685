import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from core import MessageBus
from ros_bridge import ROSBridge

load_dotenv()

# Global reference to the loop running in the Streamlit background thread
_loop = None

def set_global_loop(loop):
    global _loop
    _loop = loop

def create_agent(bus: MessageBus):
    """Creates a ReAct agent with ROS tools."""
    bridge = ROSBridge(bus)
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def async_wrap(func):
        """Wrapper to run async bridge methods inside sync LangChain tools."""
        def wrapper(*args, **kwargs):
            if _loop is None or not _loop.is_running():
                 # Fallback for testing without the streamlit loop
                return asyncio.run(func(*args, **kwargs))
            
            future = asyncio.run_coroutine_threadsafe(func(*args, **kwargs), _loop)
            return future.result()
        return wrapper

    # Wrapper functions for no-arg tools that ignore any input
    def list_nodes_wrapper(*_args, **_kwargs):
        """Wrapper that ignores input and calls list_nodes."""
        return async_wrap(bridge.list_nodes)()
    
    def list_topics_wrapper(*_args, **_kwargs):
        """Wrapper that ignores input and calls list_topics."""
        return async_wrap(bridge.list_topics)()
    
    # Wrapper functions for tools with parameters
    def get_topic_info_wrapper(topic: str):
        """Wrapper for get_topic_info."""
        return async_wrap(bridge.get_topic_info)(topic)
    
    def echo_topic_wrapper(topic: str):
        """Wrapper for echo_topic."""
        return async_wrap(bridge.echo_topic)(topic)
    
    def subscribe_topic_wrapper(topic: str):
        """Wrapper for subscribe_topic that listens to a topic once and returns the message."""
        return async_wrap(bridge.subscribe_topic)(topic)

    # Define tools using the wrapper
    tools = [
        StructuredTool.from_function(
            func=list_nodes_wrapper,
            name="list_nodes",
            description="List all active ROS nodes. Takes no arguments."
        ),
        StructuredTool.from_function(
            func=list_topics_wrapper,
            name="list_topics",
            description="List all active ROS topics. Takes no arguments."
        ),
        StructuredTool.from_function(
            func=get_topic_info_wrapper,
            name="topic_info",
            description="Get detailed info (publishers/subscribers) for a specific topic."
        ),
        StructuredTool.from_function(
            func=echo_topic_wrapper,
            name="echo_topic",
            description="Read the latest message from a topic (echo once)."
        ),
        StructuredTool.from_function(
            func=subscribe_topic_wrapper,
            name="subscribe_topic",
            description="Listen to a topic once and return the latest message. Takes topic name as input."
        )
    ]

    # Simple ReAct Prompt
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_execution_time=60
    )
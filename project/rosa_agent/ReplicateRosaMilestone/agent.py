import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from message_bus import MessageBus
from agent_interface import Agent, LangChainAgentAdapter
from ros_bridge import ROSBridge

load_dotenv()

# Global reference to the loop running in the Streamlit background thread
_loop = None

def set_global_loop(loop):
    global _loop
    _loop = loop

def create_agent(bus: MessageBus) -> Agent:
    """Creates a ReAct agent with ROS tools (decoupled from LangChain via Agent interface)."""
    bridge = ROSBridge(bus)
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def async_wrap(func):
        """Wrapper to run async bridge methods inside sync LangChain tools."""
        def wrapper(*args, **kwargs):
            if _loop is None or not _loop.is_running():
                return asyncio.run(func(*args, **kwargs))
            
            future = asyncio.run_coroutine_threadsafe(func(*args, **kwargs), _loop)
            return future.result()
        return wrapper

    # Wrapper functions for no-arg tools
    def list_nodes_wrapper(*_args, **_kwargs):
        return async_wrap(bridge.list_nodes)()
    
    def list_topics_wrapper(*_args, **_kwargs):
        return async_wrap(bridge.list_topics)()

    # Wrapper functions for tools with parameters
    def get_topic_info_wrapper(topic: str):
        return async_wrap(bridge.get_topic_info)(topic)
    
    def get_topic_publishers_wrapper(topic: str):
        return async_wrap(bridge.get_topic_publishers)(topic)
    
    def get_topic_subscribers_wrapper(topic: str):
        return async_wrap(bridge.get_topic_subscribers)(topic)
    
    def echo_topic_wrapper(topic: str):
        return async_wrap(bridge.echo_topic)(topic)
    
    def subscribe_topic_wrapper(topic: str):
        return async_wrap(bridge.subscribe_topic)(topic)

    # Define tools
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
            description="Get general information about a topic (type, publisher count, subscription count)."
        ),
        StructuredTool.from_function(
            func=get_topic_publishers_wrapper,
            name="topic_publishers",
            description="Get list of nodes that publish to a topic. Use when asked 'who publishes to this topic'."
        ),
        StructuredTool.from_function(
            func=get_topic_subscribers_wrapper,
            name="topic_subscribers",
            description="Get list of nodes that subscribe to a topic. Use when asked 'who subscribes to this topic'."
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
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_execution_time=60
    )
    
    # Wrap in abstraction layer (decoupled from LangChain)
    return LangChainAgentAdapter(agent_executor, tools)

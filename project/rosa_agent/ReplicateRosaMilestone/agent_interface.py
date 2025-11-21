"""
Agent abstraction interface to decouple from specific agent frameworks.
This allows swapping between LangChain, AutoGPT, or other frameworks.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class AgentTool(ABC):
    """Abstract interface for agent tools."""
    
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass


class Agent(ABC):
    """Abstract agent interface, decoupled from specific frameworks."""
    
    @abstractmethod
    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool to the agent."""
        pass
    
    @abstractmethod
    def process(self, user_input: str) -> str:
        """Process user input and return response."""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[AgentTool]:
        """Get list of available tools."""
        pass


class LangChainAgentAdapter(Agent):
    """Adapter to wrap LangChain agent behind the abstract interface."""
    
    def __init__(self, langchain_agent_executor, langchain_tools):
        self.agent_executor = langchain_agent_executor
        self._tools = langchain_tools
        self._agent_tools = []
    
    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool (wraps LangChain tool)."""
        self._agent_tools.append(tool)
        # Note: In a full implementation, this would also add to LangChain tools
    
    def process(self, user_input: str) -> str:
        """Process user input using LangChain agent."""
        try:
            # Invoke with input - memory is handled automatically by AgentExecutor
            response = self.agent_executor.invoke({"input": user_input})
            return response.get('output', 'No response generated')
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_tools(self) -> List[AgentTool]:
        """Get list of available tools."""
        return self._agent_tools


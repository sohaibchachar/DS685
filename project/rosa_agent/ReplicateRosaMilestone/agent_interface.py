from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class AgentTool(ABC):
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass


class Agent(ABC):
    
    @abstractmethod
    def add_tool(self, tool: AgentTool) -> None:
        pass
    
    @abstractmethod
    def process(self, user_input: str) -> str:
        pass
    
    @abstractmethod
    def get_tools(self) -> List[AgentTool]:
        pass


class LangChainAgentAdapter(Agent):
    
    def __init__(self, langchain_agent_executor, langchain_tools):
        self.agent_executor = langchain_agent_executor
        self._tools = langchain_tools
        self._agent_tools = []
    
    def add_tool(self, tool: AgentTool) -> None:
        self._agent_tools.append(tool)
    
    def process(self, user_input: str) -> str:
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response.get('output', 'No response generated')
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_tools(self) -> List[AgentTool]:
        return self._agent_tools


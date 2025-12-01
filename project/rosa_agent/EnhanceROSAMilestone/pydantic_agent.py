import os
import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from dotenv import load_dotenv

from message_bus import NATSMessageBus, MessageBus
from ros_bridge import ROSBridge

env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class NavigateToPoseRequest(BaseModel):
    x: float = Field(description="X coordinate in meters")
    y: float = Field(description="Y coordinate in meters")
    theta: float = Field(default=0.0, description="Orientation in radians (0.0 = facing positive X)")
    frame_id: str = Field(default="map", description="Frame ID for the pose")


class NavigateToPoseResponse(BaseModel):
    success: bool
    message: str
    final_pose: Optional[Dict[str, float]] = None


class GetRobotPoseResponse(BaseModel):
    x: float
    y: float
    theta: float
    frame_id: str


class SemanticLocationRequest(BaseModel):
    semantic_name: str = Field(description="Semantic name like 'table', 'bench'")


class SemanticLocationResponse(BaseModel):
    found: bool
    semantic_name: str
    x: float
    y: float
    theta: float
    description: Optional[str] = None


class SemanticLocationData(BaseModel):
    x: float
    y: float
    theta: float
    description: Optional[str] = None


class MapInfoResponse(BaseModel):
    map_name: str
    semantic_locations: Dict[str, SemanticLocationData]
    description: Optional[str] = None


class PydanticROSAgent:
    
    def __init__(self, message_bus: MessageBus, ros_bridge: ROSBridge):
        self.message_bus = message_bus
        self.ros_bridge = ros_bridge
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment. Please set it in your .env file.")
        

        instructions = """You are a ROSA (ROS Agent) assistant that helps navigate a robot in a maze environment.
You can reason about the maze layout, find semantic locations (like tables, benches, doors), and navigate the robot to specific goals.

Your capabilities:
1. Navigate to absolute coordinates (x, y, theta)
2. Find semantic locations (e.g., "go in front of a table", "navigate to the bench")
3. Get current robot pose
4. Get map information including available semantic locations

When the user asks to go to a semantic location:
1. First, get map info to see available semantic locations
2. Find the semantic location coordinates
3. Navigate to a position in front of/near the location (consider orientation)
4. Report success or any issues

Always reason step-by-step about the maze layout and navigation path before executing commands.
"""
        
        # Create agent with ROSBridge as dependency
        # Use AgentDeps to inject ros_bridge into tools
        class AgentDeps(BaseModel):
            ros_bridge: ROSBridge
            
            model_config = {
                "arbitrary_types_allowed": True  # Allow non-Pydantic types like ROSBridge
            }
        

        self.agent = Agent(
            'openai:gpt-4o',
            instructions=instructions,
            deps_type=AgentDeps,
        )

        @self.agent.tool
        async def navigate_to_pose(ctx: RunContext[AgentDeps], req: NavigateToPoseRequest) -> NavigateToPoseResponse:
            result = await ctx.deps.ros_bridge.navigate_to_pose(req.x, req.y, req.theta, req.frame_id)
            if result.get("status") == "ok":
                return NavigateToPoseResponse(
                    success=True,
                    message=result.get("data", "Navigation completed successfully"),
                    final_pose={"x": req.x, "y": req.y, "theta": req.theta}
                )
            return NavigateToPoseResponse(success=False, message=result.get("data", "Navigation failed"))
        
        @self.agent.tool
        async def get_robot_pose(ctx: RunContext[AgentDeps]) -> GetRobotPoseResponse:
            result = await ctx.deps.ros_bridge.get_robot_pose()
            if result.get("status") == "ok":
                data = result.get("data", {})
                return GetRobotPoseResponse(
                    x=data.get("x", 0.0),
                    y=data.get("y", 0.0),
                    theta=data.get("theta", 0.0),
                    frame_id=data.get("frame_id", "map")
                )
            return GetRobotPoseResponse(x=0.0, y=0.0, theta=0.0, frame_id="map")
        
        @self.agent.tool
        async def get_map_info(ctx: RunContext[AgentDeps]) -> MapInfoResponse:
            result = await ctx.deps.ros_bridge.get_map_info()
            if result.get("status") == "ok":
                data = result.get("data", {})
                semantic_locs = {}
                for name, loc_data in data.get("semantic_locations", {}).items():
                    if isinstance(loc_data, dict):
                        semantic_locs[name] = SemanticLocationData(
                            x=loc_data.get("x", 0.0),
                            y=loc_data.get("y", 0.0),
                            theta=loc_data.get("theta", 0.0),
                            description=loc_data.get("description")
                        )
                return MapInfoResponse(
                    map_name=data.get("map_name", "unknown"),
                    semantic_locations=semantic_locs,
                    description=data.get("description")
                )
            return MapInfoResponse(map_name="unknown", semantic_locations={}, description="Failed to get map info")
        
        @self.agent.tool
        async def find_semantic_location(ctx: RunContext[AgentDeps], req: SemanticLocationRequest) -> SemanticLocationResponse:
            result = await ctx.deps.ros_bridge.find_semantic_location(req.semantic_name)
            if result.get("status") == "ok":
                data = result.get("data", {})
                return SemanticLocationResponse(
                    found=True,
                    semantic_name=req.semantic_name,
                    x=data.get("x", 0.0),
                    y=data.get("y", 0.0),
                    theta=data.get("theta", 0.0),
                    description=data.get("description")
                )
            return SemanticLocationResponse(
                found=False,
                semantic_name=req.semantic_name,
                x=0.0,
                y=0.0,
                theta=0.0,
                description=f"Location '{req.semantic_name}' not found"
            )
        
        # ROS inspection tools (same as ReplicateRosaMilestone)
        @self.agent.tool
        async def list_nodes(ctx: RunContext[AgentDeps]) -> str:
            return await ctx.deps.ros_bridge.list_nodes()
        
        @self.agent.tool
        async def list_topics(ctx: RunContext[AgentDeps]) -> str:
            return await ctx.deps.ros_bridge.list_topics()
        
        @self.agent.tool
        async def topic_info(ctx: RunContext[AgentDeps], topic_name: str) -> str:
            return await ctx.deps.ros_bridge.get_topic_info(topic_name)
        
        @self.agent.tool
        async def topic_publishers(ctx: RunContext[AgentDeps], topic_name: str) -> str:
            return await ctx.deps.ros_bridge.get_topic_publishers(topic_name)
        
        @self.agent.tool
        async def topic_subscribers(ctx: RunContext[AgentDeps], topic_name: str) -> str:
            return await ctx.deps.ros_bridge.get_topic_subscribers(topic_name)
        
        @self.agent.tool
        async def echo_topic(ctx: RunContext[AgentDeps], topic_name: str) -> str:
            return await ctx.deps.ros_bridge.echo_topic(topic_name)
        
        @self.agent.tool
        async def subscribe_topic(ctx: RunContext[AgentDeps], topic_name: str) -> str:
            return await ctx.deps.ros_bridge.subscribe_topic(topic_name)
        
        # Store deps for use in run methods
        self.agent_deps = AgentDeps(ros_bridge=ros_bridge)
        # Store conversation history
        self.conversation_history = []
    
    def _convert_messages(self, messages: list) -> list:
        pydantic_messages = []
        for msg in messages:
            if msg["role"] == "user":
                pydantic_messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
            elif msg["role"] == "assistant":
                pydantic_messages.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
        return pydantic_messages
    
    async def run_streaming(self, user_query: str, previous_messages: list | None = None) -> AsyncIterator[str]:
        # Convert previous messages to pydantic_ai format
        history = self._convert_messages(previous_messages or [])
        
        # Run agent with dependencies and conversation history
        result = await self.agent.run(user_query, deps=self.agent_deps, message_history=history)
        
        # Get the response text - pydantic_ai uses result.output (not result.data)
        response_text = result.output if hasattr(result, 'output') else str(result)
        
        if isinstance(response_text, str):
            response_text = response_text.replace('\\n', '\n')
        
        for char in response_text:
            yield char
            await asyncio.sleep(0.01)
    
    async def run(self, user_query: str, previous_messages: list | None = None) -> str:
        try:
            # Convert previous messages to pydantic_ai format
            history = self._convert_messages(previous_messages or [])
            
            # Run agent with dependencies and conversation history
            result = await self.agent.run(user_query, deps=self.agent_deps, message_history=history)
            
            # Extract the actual text content - pydantic_ai uses result.output (not result.data)
            response_text = result.output if hasattr(result, 'output') else str(result)
            
            if isinstance(response_text, str):
                response_text = response_text.replace('\\n', '\n')
            
            return response_text
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}\n\nThe navigation commands may have completed successfully, but there was an issue formatting the response."


async def create_pydantic_agent(message_bus: MessageBus) -> PydanticROSAgent:
    ros_bridge = ROSBridge(message_bus)
    return PydanticROSAgent(message_bus, ros_bridge)


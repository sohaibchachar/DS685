# AI Agents for Robotics

Sohaib Chachar  
12/04/2025

## Overview

This project implements ROSA (ROS Agent) - an AI-powered agent system for robotics that enables natural language interaction with ROS 2 robots. The project is organized into two milestones, each demonstrating different agent frameworks while maintaining a decoupled architecture.

## Project Structure

### ReplicateRosaMilestone
This milestone implements a ROSA agent using **LangChain** with the ReAct (Reasoning + Acting) pattern. It demonstrates the decoupled architecture where the agent framework, message bus, and ROS are separated through abstraction layers.

**Key Files:**
- `agent.py` - LangChain ReAct agent implementation with tool definitions
- `agent_interface.py` - Abstract interface for agent decoupling (Agent, AgentTool, LangChainAgentAdapter)
- `message_bus.py` - Abstract MessageBus interface and NATS implementation
- `ros_bridge.py` - Client-side proxy that forwards ROS operations via NATS
- `ros_nats_adapter.py` - Server-side adapter that executes ROS 2 CLI commands
- `streamlit_app.py` - Web interface for interacting with the agent

**Capabilities:**
- List ROS nodes and topics
- Get topic information (type, publishers, subscribers)
- Echo and subscribe to topics
- Query topic publishers and subscribers

**Demo Video:** [ReplicateRosaMilestone Demo](https://drive.google.com/drive/folders/1RPLeSFiB-w4GorBCNqVOu4jDTx_UBB9j?usp=sharing)

### EnhanceROSAMilestone
This milestone extends the ROSA agent using **pydantic_ai** framework with ReAct-style reasoning. It adds robot navigation capabilities (both absolute coordinates and semantic locations) while maintaining the same decoupled architecture.

**Key Files:**
- `pydantic_agent.py` - Pydantic AI agent with ReAct instructions and tool definitions
- `message_bus.py` - Abstract MessageBus interface and NATS implementation
- `ros_bridge.py` - Client-side proxy for ROS operations including navigation
- `ros_nats_adapter.py` - Server-side adapter with navigation and ROS inspection handlers
- `streamlit_app.py` - Web interface with conversation memory support

**Capabilities:**
- All ROS inspection features from ReplicateRosaMilestone
- Navigate robot to absolute coordinates (x, y, theta)
- Navigate to semantic locations (e.g., "go in front of the table")
- Get current robot pose
- Get map information with semantic locations
- Find semantic locations by name

**Demo:** [EnhanceROSAMilestone Demo - Robot Navigation using LLM](https://drive.google.com/drive/folders/1ZeZdgC-ydorCJ5IJbgTreCr2TB4QqKV9?usp=sharing)  
Includes video demonstration of robot navigation and screenshot showing basic commands (getting tables info from LLM and subscribing to topics).

### Communication Flow

```
User Query (Streamlit)
    ↓
Agent (LangChain/pydantic_ai)
    ↓
ROS Bridge (Client)
    ↓
NATS Message Bus
    ↓
ROS NATS Adapter (Server)
    ↓
ROS 2 CLI Commands
    ↓
Response flows back through the chain
```

### Technology Stack

**ReplicateRosaMilestone:**
- LangChain (agent framework)
- OpenAI GPT-4o (LLM)
- NATS.io (message bus)
- Streamlit (web interface)
- ROS 2 (robotics middleware)

**EnhanceROSAMilestone:**
- pydantic_ai (agent framework)
- OpenAI GPT-4o (LLM)
- NATS.io (message bus)
- Streamlit (web interface)
- ROS 2 (robotics middleware)

## Usage

### Prerequisites (Common for Both Milestones)

**In 1st terminal (ROS 2 environment):**
```bash
source /workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash
ros2 launch tb_worlds tb_demo_world.launch.py
```

### Running the Agents

Both milestones follow the same setup pattern. Use the appropriate milestone directory:

**In 2nd terminal:**
```bash
# For ReplicateRosaMilestone:
cd /workspaces/eng-ai-agents/project/rosa_agent/ReplicateRosaMilestone/

# OR for EnhanceROSAMilestone:
# cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone/

nats-server -p 4222
```

**In 3rd terminal:**
```bash
# For ReplicateRosaMilestone:
cd /workspaces/eng-ai-agents/project/rosa_agent/ReplicateRosaMilestone/

# OR for EnhanceROSAMilestone:
# cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone/

python3 ros_nats_adapter.py
```

**In 4th terminal:**
```bash
# For ReplicateRosaMilestone:
cd /workspaces/eng-ai-agents/project/rosa_agent/ReplicateRosaMilestone/

# OR for EnhanceROSAMilestone:
# cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone/

streamlit run streamlit_app.py
```

**Note:** Make sure to set `OPENAI_API_KEY` environment variable before running the agents.


# ROSA Agent - ReAct-based AI Agent for ROS2

## Overview

AI agent based on ReAct principles that replicates ROSA paper functionality for ROS2 interactions.

## Key Features

- **ReAct Pattern**: Reasoning and Acting for decision-making
- **ROS2 CLI Tools**: List nodes, topics, echo messages, custom commands, **robot movement**
- **Multiple LLM Support**: OpenAI or Anthropic
- **Streamlit UI**: Beautiful web interface (recommended)
- **LangChain Implementation**: Current stable version

## Installation

```bash
cd /workspaces/eng-ai-agents
source .venv/bin/activate
```

## Setup

Create `.env` file with your API key:

```bash
OPENAI_API_KEY=your-key-here
# OR
ANTHROPIC_API_KEY=your-key-here
```

## Usage

### Streamlit Interface (Recommended)

```bash
python -m project.rosa_agent --interface streamlit
```

Opens at `http://localhost:8501`

### CLI Interface

```bash
python -m project.rosa_agent --interface cli
```

## Example Queries

### ROS CLI Queries (ROSA Paper Style)
- "List all ROS2 nodes"
- "What topics are available?"
- "Subscribe to the /odom topic"
- "Get information about /camera/image_raw"

### Robot Movement Queries
- "Move the robot backward for 10 seconds"
- "Move robot forward at 0.2 m/s for 5 seconds"
- "Navigate the robot to location (3, 2)"
- "Move robot to coordinates 0, 0"

## Available Tools (7 total)

1. `list_nodes` - List all running ROS2 nodes
2. `list_topics` - List all ROS2 topics
3. `topic_info` - Get topic information
4. `topic_echo` - Echo messages from a topic
5. `move_robot_velocity` - Move robot with velocity commands (forward/backward/rotate)
6. `navigate_to_pose` - Navigate to specific coordinates
7. `execute_custom_command` - Execute any ROS2 CLI command

## Architecture

```
User Query → ReAct Agent (LangChain) → ROS Tools → ROS CLI Executor → ROS2 System
```

## Robot Movement

The agent can control robot movement through two methods:

1. **Velocity Commands** (`move_robot_velocity`):
   - `linear_x`: Forward (+) or backward (-) velocity in m/s
   - `angular_z`: Rotational velocity in rad/s
   - `duration`: Time in seconds to move

2. **Navigation** (`navigate_to_pose`):
   - `x, y`: Target coordinates in map frame
   - `theta`: Target orientation (optional)

## Project Status

✅ **Milestone 1**: Replicate ROSA with LangChain - COMPLETE
- ROS CLI interaction
- Robot movement tools
- Streamlit interface

⏳ **Milestone 2**: Add pydantic_ai and advanced navigation
⏳ **Future**: NATS/JetStream decoupling

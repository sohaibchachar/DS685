# Enhanced ROSA Milestone - Pydantic AI Agent

An enhanced ROSA agent using **pydantic_ai** for type-safe agent definition with navigation and reasoning capabilities.

## Features

- 🤖 **Pydantic AI Integration**: Type-safe agent definition using pydantic_ai
- 🧭 **Navigation Capabilities**: Navigate to absolute coordinates or semantic locations
- 💬 **Streaming Responses**: Real-time streaming of agent responses in Streamlit
- 🧠 **LLM Reasoning**: Uses OpenAI GPT-4o with reasoning API for maze navigation
- 🗺️ **Semantic Navigation**: Navigate to semantic locations like "table", "bench", "door"
- 📡 **NATS Message Bus**: Decoupled architecture via NATS
- 🔄 **Dual Framework Support**: Can run alongside LangChain for testing

## Architecture

```
Streamlit UI → Pydantic AI Agent → ROS Bridge → NATS Message Bus → ROS-NATS Adapter → ROS 2
```

### Components

1. **Pydantic AI Agent** (`pydantic_agent.py`): 
   - Type-safe agent with Pydantic models
   - Navigation tools with reasoning
   - Streaming support

2. **Enhanced ROS Bridge** (`ros_bridge.py`):
   - Navigation operations
   - Semantic location finding
   - Map information retrieval

3. **ROS-NATS Adapter** (`ros_nats_adapter.py`):
   - Handles navigation requests
   - Manages semantic locations
   - Bridges ROS 2 actions to NATS

4. **Streamlit Interface** (`streamlit_app.py`):
   - Streaming chat interface
   - Real-time response display
   - Configuration panel

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up `.env` file in workspace root:
```
OPENAI_API_KEY=your_api_key_here
USE_NATS=false  # Set to true if using NATS
NATS_URL=nats://localhost:4222
```

## Running the Agent

### Option 1: With NATS (Production)

1. Start NATS server:
```bash
./start_nats.sh
```

2. Start ROS-NATS adapter (in a separate terminal):
```bash
./start_ros_adapter.sh
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Option 2: Without NATS (Development/Testing)

1. Set `USE_NATS=false` in your `.env` file

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Usage Examples

### Absolute Navigation
- "Navigate to x=2.0, y=1.5"
- "Go to coordinates (1.0, 0.5) with orientation 0.0"
- "Move the robot to position x=3.0, y=2.0"

### Semantic Navigation
- "Go in front of the table"
- "Navigate to the bench"
- "Move to the door"
- "Go near the chair"

### Information Queries
- "What is my current position?"
- "What semantic locations are available?"
- "Show me the map information"

## Navigation Tools

The agent has access to the following tools:

1. **`navigate_to_pose`**: Navigate to absolute coordinates (x, y, theta)
2. **`get_robot_pose`**: Get current robot position
3. **`get_map_info`**: Get map information and semantic locations
4. **`find_semantic_location`**: Find coordinates of semantic locations

## Semantic Locations

The agent supports semantic navigation to locations like:
- `table`: A table in the center
- `bench`: A bench near the wall
- `door`: Main entrance door
- `chair`: A chair

These can be extended by modifying `semantic_locations` in `ros_nats_adapter.py`.

## Streaming Responses

The Streamlit interface supports real-time streaming of agent responses, providing immediate feedback as the agent reasons and executes navigation commands.

## Reasoning Capabilities

The agent uses OpenAI's GPT-4o with reasoning API to:
1. Understand navigation goals (absolute or semantic)
2. Plan navigation paths
3. Execute navigation commands
4. Report success or handle errors

## Integration with LangChain

Both pydantic_ai and LangChain can coexist for testing purposes. The architecture allows:
- Using pydantic_ai for navigation (primary)
- Using LangChain for other operations (if needed)
- Switching between frameworks as needed

## Troubleshooting

### Agent Not Initializing
- Check that `OPENAI_API_KEY` is set in `.env`
- Verify NATS server is running if `USE_NATS=true`
- Check error messages in Streamlit interface

### Navigation Not Working
- Ensure ROS 2 Nav2 is running
- Verify `/navigate_to_pose` action server is available
- Check ROS-NATS adapter logs for errors

### Streaming Not Working
- Ensure event loop is properly initialized
- Check that agent is using `run_streaming()` method
- Verify Streamlit version supports async operations

## Development

### Adding New Semantic Locations

Edit `ros_nats_adapter.py`:
```python
self.semantic_locations = {
    "your_location": {"x": 1.0, "y": 2.0, "theta": 0.0, "description": "..."}
}
```

### Extending Navigation Tools

Add new tools to `pydantic_agent.py`:
1. Define Pydantic request/response models
2. Create async tool method
3. Add to agent tools list

## License

Part of the ROSA Agent project.





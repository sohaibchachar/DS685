# ROSA Agent Streamlit App

A web-based interface for interacting with the decoupled ROSA agent.

## Features

- 🤖 **Interactive Chat Interface**: Chat with the ROSA agent through a web UI
- ⚙️ **Configuration Panel**: Configure NATS settings and initialize the agent
- 💬 **Conversation Memory**: The agent remembers previous conversations
- 📡 **NATS Support**: Connect to NATS message bus or use mock for development
- 🛠️ **ROS 2 Tools**: Access to list nodes, topics, get topic info, echo, and subscribe

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a `.env` file in the workspace root with your `OPENAI_API_KEY`:
```
OPENAI_API_KEY=your_api_key_here
USE_NATS=false  # Set to true if using NATS
NATS_URL=nats://localhost:4222
```

## Running the App

### Option 1: With NATS (Production)

1. Start NATS server (if not already running):
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

1. Set `USE_NATS=false` in your `.env` file or use the sidebar checkbox

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will use `MockMessageBus` for testing without a NATS server.

## Usage

1. **Initialize the Agent**: 
   - Configure NATS settings in the sidebar
   - Click "🔄 Initialize Agent" button

2. **Chat with the Agent**:
   - Type your questions in the chat input
   - Ask about ROS 2 topics, nodes, or operations
   - Examples:
     - "List all ROS topics"
     - "What nodes are running?"
     - "Get info about /camera/image_raw topic"
     - "Subscribe to /cmd_vel topic and show me 5 messages"

3. **Clear History**: 
   - Click "🗑️ Clear Chat History" in the sidebar

4. **Disconnect**: 
   - Click "🛑 Disconnect" to clean up resources

## Available Tools

The agent has access to the following ROS 2 tools:

- `list_nodes`: List all active ROS 2 nodes
- `list_topics`: List all active ROS 2 topics
- `topic_info`: Get detailed information about a specific topic
- `echo_topic`: Get a single message from a topic
- `subscribe_topic`: Subscribe to a topic and collect multiple messages

## Architecture

The Streamlit app interfaces with the decoupled ROSA agent architecture:

```
Streamlit UI → Agent Interface → ROS Bridge → NATS Message Bus → ROS-NATS Adapter → ROS 2
```

This architecture ensures:
- **Decoupling from ROS**: Communication via NATS message bus
- **Decoupling from Framework**: Agent abstraction layer allows swapping implementations
- **Web Accessibility**: Easy-to-use web interface for non-technical users

## Troubleshooting

### Agent Not Initializing

- Check that `OPENAI_API_KEY` is set in `.env` file
- Verify NATS server is running if `USE_NATS=true`
- Check the error messages in the Streamlit interface

### NATS Connection Issues

- Ensure NATS server is running: `./start_nats.sh`
- Verify `NATS_URL` is correct (default: `nats://localhost:4222`)
- Check that ROS-NATS adapter is running: `./start_ros_adapter.sh`

### No Responses from Agent

- Make sure the agent is initialized (green status indicator)
- Check that ROS 2 is running and has active topics/nodes
- Review the agent's verbose output in the terminal where Streamlit is running





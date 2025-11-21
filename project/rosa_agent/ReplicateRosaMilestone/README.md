# ROSA Agent - Decoupled Architecture

A decoupled AI agent for ROS 2 interactions using NATS.io message bus and agent abstraction.

## Architecture Principles

This implementation follows two key decoupling principles:

1. **Decoupled from ROS Simulator**: The AI agent communicates with ROS via NATS.io/JetStream message bus, not direct ROS calls. This allows swapping simulators without changing the agent.

2. **Decoupled from Agent Framework**: The agent uses an abstraction layer (`Agent` interface) that allows swapping between LangChain, AutoGPT, or other frameworks without changing the core logic.

## Architecture Components

```
┌─────────────────┐         ┌──────────────┐         ┌─────────────┐
│   AI Agent      │◄────────┤  Message Bus │◄────────┤ ROS Adapter │
│  (LangChain)    │  NATS   │  (NATS.io)   │  NATS   │  (ROS 2)    │
└─────────────────┘         └──────────────┘         └─────────────┘
       ▲
       │ Agent Interface (abstraction)
       │
┌─────────────────┐
│ Other Frameworks│
│ (AutoGPT, etc.)│
└─────────────────┘
```

### Components

- **`message_bus.py`**: NATS.io message passing layer (can be swapped for other message buses)
- **`agent_interface.py`**: Abstract agent interface (decouples from LangChain)
- **`ros_bridge.py`**: ROS operations via message bus (decouples from direct ROS calls)
- **`ros_nats_adapter.py`**: ROS-to-NATS bridge service (runs separately)
- **`agent.py`**: Main agent using decoupled architecture

## Setup

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export USE_NATS="true"  # Set to false to use mock message bus
   export NATS_URL="nats://localhost:4222"  # NATS server URL
   ```

4. **Start NATS server** (if using NATS):
   ```bash
   # Using Docker
   docker run -d --name nats-server -p 4222:4222 nats:latest
   
   # Or install locally: https://docs.nats.io/running-a-nats-service/introduction/installation
   ```

5. **Start ROS-NATS adapter** (in separate terminal):
   ```bash
   source /opt/ros/jazzy/setup.bash
   python3 ros_nats_adapter.py
   ```

6. **Run the agent:**
   ```bash
   python3 agent.py
   ```

## Usage

### With NATS (Production)

1. Start NATS server
2. Start ROS-NATS adapter
3. Run agent with `USE_NATS=true`

### Without NATS (Development/Testing)

Run agent with `USE_NATS=false` (uses MockMessageBus)

## Example Queries

- "List all active nodes"
- "What topics are available?"
- "Show me the data on /clock topic"
- "Get info about /cmd_vel topic"

Type `quit` or `exit` to stop.

## Decoupling Benefits

1. **ROS Decoupling**: Agent doesn't need ROS installed - only needs NATS connection
2. **Framework Decoupling**: Can swap LangChain for other frameworks by implementing `Agent` interface
3. **Message Bus Decoupling**: Can swap NATS for RabbitMQ, Kafka, etc. by implementing `MessageBus` interface
4. **Testing**: Use `MockMessageBus` for testing without ROS or NATS

## Tools Available

- `list_nodes`: Lists all active ROS 2 nodes (via message bus)
- `list_topics`: Lists all active ROS 2 topics (via message bus)
- `topic_info`: Get information about a specific topic (via message bus)
- `echo_topic`: Echo one message from a topic (via message bus)

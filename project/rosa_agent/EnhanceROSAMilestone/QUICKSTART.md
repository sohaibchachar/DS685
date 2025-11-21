# Quick Start Guide - EnhanceROSAMilestone

## Prerequisites

1. **Python Virtual Environment**: Make sure you have a virtual environment activated
2. **ROS 2**: ROS 2 Jazzy should be installed and sourced
3. **Dependencies**: Install required packages
4. **Environment Variables**: Set up `.env` file

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file in the workspace root (`/workspaces/eng-ai-agents/.env`) contains:

```env
OPENAI_API_KEY=your_api_key_here
USE_NATS=true  # or false for mock mode
NATS_URL=nats://localhost:4222
```

### 3. Start ROS 2 Simulation (if needed)

If you want to test navigation, start your ROS 2 simulation:

```bash
# In a separate terminal
ros2 launch tb_worlds tb_demo_world.launch.py
```

Wait for Nav2 to fully initialize (this may take 10-30 seconds).

## Running the Agent

### Option A: With NATS (Recommended for Production)

**Terminal 1 - Start NATS Server:**
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone
chmod +x start_nats.sh
./start_nats.sh
```

**Terminal 2 - Start ROS-NATS Adapter:**
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone
chmod +x start_ros_adapter.sh
./start_ros_adapter.sh
```

**Terminal 3 - Start Streamlit App:**
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone
chmod +x start_streamlit.sh
./start_streamlit.sh
```

Then open your browser to: **http://localhost:8501**

### Option B: Without NATS (Mock Mode for Testing)

If you set `USE_NATS=false` in your `.env` file, you can run just the Streamlit app:

```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/EnhanceROSAMilestone
chmod +x start_streamlit.sh
./start_streamlit.sh
```

This uses a mock message bus and doesn't require NATS or ROS 2.

## Usage

Once the Streamlit app is running:

1. **Initialize the Agent**: Click "Initialize Agent" in the sidebar
2. **Start Chatting**: Type navigation commands in the chat input

### Example Commands:

**Absolute Navigation:**
- "Navigate to x=2.0, y=3.5"
- "Go to coordinates (1.0, 0.5) with orientation 0.0"

**Semantic Navigation:**
- "Go in front of the table"
- "Navigate to the bench"
- "Move to the door"

**Information Queries:**
- "What is my current position?"
- "What semantic locations are available?"
- "Show me the map information"

## Troubleshooting

### NATS Server Won't Start
- Check if port 4222 is already in use: `lsof -i :4222`
- Kill existing NATS server: `pkill nats-server`

### ROS-NATS Adapter Errors
- Ensure ROS 2 is sourced: `source /opt/ros/jazzy/setup.bash`
- Check that ROS 2 is running: `ros2 node list`

### Streamlit App Won't Initialize
- Verify `OPENAI_API_KEY` is set in `.env`
- Check that dependencies are installed: `pip list | grep pydantic-ai`
- Look for error messages in the Streamlit interface

### Navigation Goals Rejected
- Ensure Nav2 is fully initialized (wait 10-30 seconds after launching)
- Set robot initial pose in RViz
- Check that `/navigate_to_pose` action server is active: `ros2 action list`

## File Structure

```
EnhanceROSAMilestone/
├── pydantic_agent.py      # Main Pydantic AI agent
├── ros_bridge.py          # ROS Bridge for navigation
├── ros_nats_adapter.py    # ROS-NATS adapter service
├── message_bus.py         # NATS message bus abstraction
├── streamlit_app.py       # Streamlit web interface
├── start_nats.sh          # Start NATS server
├── start_ros_adapter.sh   # Start ROS-NATS adapter
├── start_streamlit.sh     # Start Streamlit app
├── requirements.txt       # Python dependencies
└── README.md             # Detailed documentation
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check semantic locations in `ros_nats_adapter.py`
- Customize navigation tools in `pydantic_agent.py`



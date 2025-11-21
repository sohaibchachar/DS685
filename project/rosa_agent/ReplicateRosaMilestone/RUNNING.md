# Running ROSA Agent with NATS (Inside Docker)

Since you're running inside a Docker container, you need to run NATS server directly (not via Docker).

## Quick Start

You need **3 terminals** (or use `tmux`/`screen` for multiple sessions):

### Terminal 1: Start NATS Server
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent
./start_nats.sh
```
Or manually:
```bash
nats-server -p 4222
```

### Terminal 2: Start ROS-NATS Adapter
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent
./start_ros_adapter.sh
```
Or manually:
```bash
source ../../.venv/bin/activate
source /opt/ros/jazzy/setup.bash  # if ROS is available
export NATS_URL="nats://localhost:4222"
python3 ros_nats_adapter.py
```

### Terminal 3: Start the Agent
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent
./start_agent.sh
```
Or manually:
```bash
source ../../.venv/bin/activate
export USE_NATS=true
export NATS_URL="nats://localhost:4222"
python3 agent.py
```

## Using tmux (Single Terminal)

If you only have one terminal, use `tmux`:

```bash
# Install tmux if needed
sudo apt-get update && sudo apt-get install -y tmux

# Start tmux session
tmux new -s rosa

# Split into 3 panes (Ctrl+b then % for vertical, " for horizontal)
# Or create new windows: Ctrl+b then c
# Switch windows: Ctrl+b then 0, 1, 2

# In pane 1: Start NATS
./start_nats.sh

# In pane 2: Start ROS adapter
./start_ros_adapter.sh

# In pane 3: Start agent
./start_agent.sh
```

## Using Background Processes

Alternatively, run NATS and adapter in background:

```bash
# Terminal 1: Start NATS in background
cd /workspaces/eng-ai-agents/project/rosa_agent
nats-server -p 4222 > /tmp/nats.log 2>&1 &
NATS_PID=$!
echo "NATS server started (PID: $NATS_PID)"

# Terminal 2: Start ROS adapter in background
source ../../.venv/bin/activate
source /opt/ros/jazzy/setup.bash  # if available
export NATS_URL="nats://localhost:4222"
python3 ros_nats_adapter.py > /tmp/ros_adapter.log 2>&1 &
ADAPTER_PID=$!
echo "ROS adapter started (PID: $ADAPTER_PID)"

# Terminal 3: Run agent (foreground)
source ../../.venv/bin/activate
export USE_NATS=true
export NATS_URL="nats://localhost:4222"
python3 agent.py

# To stop background processes:
# kill $NATS_PID $ADAPTER_PID
```

## Verify NATS is Running

```bash
# Check if NATS is listening
netstat -tlnp | grep 4222
# or
ss -tlnp | grep 4222

# Test NATS connection
python3 -c "import nats; import asyncio; asyncio.run(nats.connect('nats://localhost:4222'))"
```

## Troubleshooting

1. **NATS not starting**: Check if port 4222 is already in use
   ```bash
   lsof -i :4222
   ```

2. **Connection refused**: Make sure NATS server is running before starting adapter/agent

3. **ROS not found**: The ROS adapter will still work using CLI commands even if ROS isn't fully sourced

4. **Check logs**: 
   - NATS: Check terminal output or `/tmp/nats.log`
   - ROS adapter: Check terminal output or `/tmp/ros_adapter.log`


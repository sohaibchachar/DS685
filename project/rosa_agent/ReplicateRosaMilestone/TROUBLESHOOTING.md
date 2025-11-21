# Troubleshooting Guide

## NATS Timeout Errors

If you see errors like `Error listing topics: nats: timeout`, it means:

1. **NATS server is not running**, OR
2. **ROS-NATS adapter is not running**

### Solution 1: Use MockMessageBus (Quick Testing)

If you just want to test the agent without setting up NATS:

1. **In Streamlit app**: Uncheck "Use NATS Message Bus" checkbox in the sidebar
2. **Or set environment variable**: `USE_NATS=false` before running

This will use `MockMessageBus` which doesn't require NATS or ROS 2.

### Solution 2: Set Up NATS Properly (For Real ROS 2 Interaction)

To use NATS with real ROS 2, you need **3 things running**:

#### Step 1: Start NATS Server
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/ReplicateRosaMilestone
./start_nats.sh
```

You should see:
```
Listening for client connections on nats://0.0.0.0:4222
```

#### Step 2: Start ROS-NATS Adapter (in a NEW terminal)
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/ReplicateRosaMilestone
./start_ros_adapter.sh
```

You should see:
```
Subscribing to ros.nodes.list...
Subscribing to ros.topics.list...
Subscribing to ros.topic.info...
Subscribing to ros.topic.echo...
Subscribing to ros.topic.subscribe...
All handlers subscribed successfully!
ROS-NATS Adapter running. Listening for requests...
```

#### Step 3: Start Streamlit App (in a NEW terminal)
```bash
cd /workspaces/eng-ai-agents/project/rosa_agent/ReplicateRosaMilestone
streamlit run streamlit_app.py
```

Then in the Streamlit UI:
- ✅ Check "Use NATS Message Bus"
- Enter NATS URL: `nats://localhost:4222`
- Click "🔄 Initialize Agent"

### Verifying NATS Connection

You can test if NATS is running:
```bash
# Check if NATS server is accessible
nc -zv localhost 4222
```

Or use the NATS CLI:
```bash
nats server check
```

### Common Issues

1. **"nats: timeout" error**
   - NATS server not running → Start with `./start_nats.sh`
   - ROS-NATS adapter not running → Start with `./start_ros_adapter.sh`
   - Wrong NATS URL → Check it's `nats://localhost:4222`

2. **"Not connected to NATS" error**
   - Agent not initialized → Click "🔄 Initialize Agent" in Streamlit
   - Connection failed → Check NATS server is running

3. **"Unknown topic" or "No nodes found"**
   - ROS 2 not running → Make sure ROS 2 is set up and running
   - ROS-NATS adapter not connected to ROS 2 → Check adapter terminal for errors

### Quick Test Without NATS

For quick testing, use MockMessageBus:

```python
# In Python
import os
os.environ["USE_NATS"] = "false"

# Then run your agent
```

Or in Streamlit: Just uncheck the "Use NATS Message Bus" checkbox.





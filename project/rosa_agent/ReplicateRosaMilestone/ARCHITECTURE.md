# Decoupled Architecture Documentation

## Overview

This ROSA agent implementation follows strict decoupling principles to ensure flexibility and maintainability.

## Decoupling Principles

### 1. Decoupled from ROS Simulator

**Problem**: Original implementation directly called ROS 2 CLI commands via `subprocess`, tightly coupling the agent to ROS.

**Solution**: 
- All ROS interactions go through a NATS.io message bus
- `ros_bridge.py` provides ROS operations via message bus requests
- `ros_nats_adapter.py` runs as a separate service that bridges ROS and NATS
- Agent never directly calls ROS commands

**Benefits**:
- Agent can run without ROS installed
- Can swap ROS for other simulators (Gazebo, Webots, etc.) by changing the adapter
- Agent and simulator can run on different machines
- Easy to test with mock message bus

### 2. Decoupled from Agent Framework

**Problem**: Original implementation was tightly coupled to LangChain (`AgentExecutor`, `create_react_agent`, etc.).

**Solution**:
- `agent_interface.py` defines abstract `Agent` and `AgentTool` interfaces
- `LangChainAgentAdapter` wraps LangChain behind the abstraction
- Main agent code uses the `Agent` interface, not LangChain directly

**Benefits**:
- Can swap LangChain for AutoGPT, CrewAI, or custom frameworks
- Only need to implement `Agent` interface to add new framework
- Core agent logic remains unchanged when swapping frameworks

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Agent Abstraction Layer                     │
│  (agent_interface.py - Agent, AgentTool interfaces)     │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ LangChain   │  │  AutoGPT    │  │   Custom    │
│  Adapter    │  │  Adapter    │  │  Adapter    │
└─────────────┘  └─────────────┘  └─────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Message Bus Abstraction                     │
│  (message_bus.py - MessageBus interface)                │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              NATS Message Bus                           │
│  (NATSMessageBus - NATS.io implementation)             │
└─────────────────────────────────────────────────────────┘
         │
         │ NATS Protocol
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              ROS-NATS Adapter Service                   │
│  (ros_nats_adapter.py - separate process)               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              ROS 2 Simulator                            │
└─────────────────────────────────────────────────────────┘
```

## File Structure

- **`message_bus.py`**: Message bus abstraction and NATS implementation
- **`agent_interface.py`**: Agent framework abstraction
- **`ros_bridge.py`**: ROS operations via message bus (used by agent)
- **`ros_nats_adapter.py`**: ROS-to-NATS bridge service (separate process)
- **`agent.py`**: Main agent using decoupled architecture

## Swapping Components

### Swap Agent Framework

To use a different agent framework (e.g., AutoGPT):

1. Implement `Agent` interface in `agent_interface.py`
2. Create adapter class (e.g., `AutoGPTAgentAdapter`)
3. Update `create_agent()` in `agent.py` to use new adapter

### Swap Message Bus

To use a different message bus (e.g., RabbitMQ):

1. Implement `MessageBus` interface in `message_bus.py`
2. Create implementation class (e.g., `RabbitMQMessageBus`)
3. Update `main_async()` in `agent.py` to use new message bus

### Swap Simulator

To use a different simulator (e.g., Gazebo):

1. Create new adapter (e.g., `gazebo_nats_adapter.py`)
2. Implement same NATS subjects as `ros_nats_adapter.py`
3. Agent code requires no changes

## Testing

Use `MockMessageBus` for testing without ROS or NATS:

```python
message_bus = MockMessageBus()
agent = create_agent(message_bus)
```

## Benefits Summary

1. **Modularity**: Each component can be swapped independently
2. **Testability**: Can test agent without ROS or NATS
3. **Scalability**: Agent and simulator can run on different machines
4. **Flexibility**: Easy to add new frameworks or simulators
5. **Maintainability**: Clear separation of concerns


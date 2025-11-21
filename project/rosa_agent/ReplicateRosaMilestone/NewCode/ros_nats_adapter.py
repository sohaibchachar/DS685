import asyncio
import os
import subprocess
from core import MessageBus

# Map NATS subjects to ROS 2 CLI commands
COMMANDS = {
    "ros.nodes.list": ["ros2", "node", "list"],
    "ros.topics.list": ["ros2", "topic", "list"],
    # Lambdas handle dynamic arguments from the message payload
    "ros.topic.info": lambda data: ["ros2", "topic", "info", data.get("topic"), "--verbose"],
    "ros.topic.echo": lambda data: ["ros2", "topic", "echo", data.get("topic"), "--once"],
}

async def execute_command(cmd_args):
    """Executes a shell command asynchronously and captures output."""
    if not cmd_args:
        return "Error: Invalid command configuration."
        
    try:
        print(f"Executing: {' '.join(cmd_args)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        output = stdout.decode().strip()
        error = stderr.decode().strip()
        
        if error and not output:
            return f"Command Error: {error}"
        return output if output else "Command executed (no output)."
    except Exception as e:
        return f"System Execution Error: {str(e)}"

async def run_adapter():
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    bus = MessageBus(nats_url)
    try:
        await bus.connect()
    except Exception as e:
        print(f"Failed to connect to NATS at {nats_url}: {e}")
        return

    # Dynamic handler generator
    async def make_handler(cmd_def):
        return lambda data: execute_command(cmd_def(data) if callable(cmd_def) else cmd_def)

    # 1. Subscribe to standard commands defined in COMMANDS
    for subject, cmd_def in COMMANDS.items():
        print(f"Subscribing to {subject}...")
        await bus.subscribe(subject, await make_handler(cmd_def))

    # 2. Special Case: Subscribe to topic (listen once)
    async def handle_subscribe(data):
        topic = data.get("topic")
        # Use --once to get just one message
        cmd = ["ros2", "topic", "echo", topic, "--once"]
        return await execute_command(cmd)

    print(f"Subscribing to ros.topic.subscribe...")
    await bus.subscribe("ros.topic.subscribe", handle_subscribe)

    print("ROS-NATS Adapter Running... (Press Ctrl+C to stop)")
    # Keep the event loop running forever
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(run_adapter())
    except KeyboardInterrupt:
        print("\nAdapter stopped.")
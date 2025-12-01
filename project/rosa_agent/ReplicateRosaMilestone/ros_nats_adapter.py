import asyncio
import os
import subprocess
from message_bus import NATSMessageBus


COMMANDS = {
    "ros.nodes.list": ["ros2", "node", "list"],
    "ros.topics.list": ["ros2", "topic", "list"],
    "ros.topic.info": lambda data: ["ros2", "topic", "info", data.get("topic"), "--verbose"],
    "ros.topic.echo": lambda data: ["ros2", "topic", "echo", data.get("topic"), "--once"],
}

def _parse_topic_info(output: str):
    lines = output.strip().split('\n')
    info = {
        "type": None,
        "publisher_count": 0,
        "subscription_count": 0,
        "publishers": [],
        "subscribers": []
    }
    
    def _finalize_current_node():
        if current_node_name and current_section:
            if current_namespace and current_namespace != "/":
                full_name = current_namespace.rstrip("/") + "/" + current_node_name.lstrip("/")
            else:
                full_name = "/" + current_node_name.lstrip("/")
            
            if current_section == "publishers" and full_name not in info["publishers"]:
                info["publishers"].append(full_name)
            elif current_section == "subscribers" and full_name not in info["subscribers"]:
                info["subscribers"].append(full_name)
    
    current_section = None
    current_node_name = None
    current_namespace = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Type:"):
            info["type"] = line.split("Type:")[1].strip()
        elif "Publisher count:" in line or "publisher count:" in line.lower():
            _finalize_current_node()
            current_node_name = None
            current_namespace = None
            try:
                count_str = line.split(":")[1].strip()
                info["publisher_count"] = int(count_str)
                current_section = "publishers"
            except:
                pass
        elif "Subscription count:" in line or "subscription count:" in line.lower():
            _finalize_current_node()
            current_node_name = None
            current_namespace = None
            try:
                count_str = line.split(":")[1].strip()
                info["subscription_count"] = int(count_str)
                current_section = "subscribers"
            except:
                pass
        elif line.startswith("Node name:"):
            current_node_name = line.split("Node name:")[1].strip()
        elif line.startswith("Node namespace:"):
            current_namespace = line.split("Node namespace:")[1].strip()
            if current_node_name:
                _finalize_current_node()
                current_node_name = None
                current_namespace = None
        elif line.startswith("/") and current_section and not line.startswith("/ "):
            if current_section == "publishers" and line not in info["publishers"]:
                info["publishers"].append(line)
            elif current_section == "subscribers" and line not in info["subscribers"]:
                info["subscribers"].append(line)
    
    _finalize_current_node()
    return info

async def execute_command(cmd_args):
    if not cmd_args:
        return "Error: Invalid command configuration."
        
    try:
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
    bus = NATSMessageBus(nats_url)
    try:
        await bus.connect()
    except Exception as e:
        print(f"Failed to connect to NATS at {nats_url}: {e}")
        return


    async def make_handler(cmd_def):
        return lambda data: execute_command(cmd_def(data) if callable(cmd_def) else cmd_def)

    for subject, cmd_def in COMMANDS.items():
        print(f"Subscribing to {subject}...")
        await bus.subscribe(subject, await make_handler(cmd_def))


    async def handle_topic_info(data):
        topic = data.get("topic")
        cmd = ["ros2", "topic", "info", topic, "--verbose"]
        output = await execute_command(cmd)
        parsed = _parse_topic_info(output)
        general_info = f"Type: {parsed['type']}\n"
        general_info += f"Publisher count: {parsed['publisher_count']}\n"
        general_info += f"Subscription count: {parsed['subscription_count']}"
        return general_info

    async def handle_topic_publishers(data):
        topic = data.get("topic")
        cmd = ["ros2", "topic", "info", topic, "--verbose"]
        output = await execute_command(cmd)
        parsed = _parse_topic_info(output)
        if parsed["publishers"]:
            publishers_list = "\n".join([f"  - {pub}" for pub in parsed["publishers"]])
            return f"Publishers for {topic} ({parsed['publisher_count']}):\n{publishers_list}"
        return f"No publishers found for {topic}"

    async def handle_topic_subscribers(data):
        topic = data.get("topic")
        cmd = ["ros2", "topic", "info", topic, "--verbose"]
        output = await execute_command(cmd)
        parsed = _parse_topic_info(output)
        if parsed["subscribers"]:
            subscribers_list = "\n".join([f"  - {sub}" for sub in parsed["subscribers"]])
            return f"Subscribers for {topic} ({parsed['subscription_count']}):\n{subscribers_list}"
        return f"No subscribers found for {topic}"

    async def handle_subscribe(data):
        topic = data.get("topic")
        cmd = ["ros2", "topic", "echo", topic, "--once"]
        return await execute_command(cmd)

    print("Subscribing to ros.topic.info...")
    await bus.subscribe("ros.topic.info", handle_topic_info)
    print("Subscribing to ros.topic.publishers...")
    await bus.subscribe("ros.topic.publishers", handle_topic_publishers)
    print("Subscribing to ros.topic.subscribers...")
    await bus.subscribe("ros.topic.subscribers", handle_topic_subscribers)
    print("Subscribing to ros.topic.subscribe...")
    await bus.subscribe("ros.topic.subscribe", handle_subscribe)

    print("ROS-NATS Adapter Running... (Press Ctrl+C to stop)")
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(run_adapter())
    except KeyboardInterrupt:
        print("\nAdapter stopped.")

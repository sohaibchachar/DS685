import asyncio
import os
import subprocess
import re
import math
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

# Semantic locations based on actual items in sim_house.sdf.xacro world file
semantic_locations = {
    "platform_1": {"x": -0.355, "y": 0.8870, "theta": 2.724, "description": "Detection table 1 with banana and backpack"},
    "platform_2": {"x": 2.89, "y": 3.34, "theta": -3.11, "description": "Detection table 2 with book and water bottle"},
    "platform_3": {"x": 5.19, "y": 2.80, "theta": 1.57, "description": "Detection table 3 with birthday cake and ball"},
    "banana": {"x": -0.355, "y": 0.8870, "theta": 2.724, "description": "Banana for scale on table 1"},
    "backpack": {"x": -0.355, "y": 0.8870, "theta": 2.724, "description": "JanSport backpack red on table 1"},
    "book": {"x": 2.89, "y": 3.34, "theta": -3.11, "description": "Eat to Live book on table 2"},
    "water_bottle": {"x": 2.89, "y": 3.34, "theta": -3.11, "description": "Water bottle on table 2"},
    "cake": {"x": 5.19, "y": 2.95, "theta": 0.0, "description": "Birthday cake on table 3"},
    "ball": {"x": 5.19, "y": 2.65, "theta": 0.0, "description": "RoboCup 3D simulation ball on table 3"},
}

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

async def handle_navigate_to_pose(data):
    x = data.get("x", 0.0)
    y = data.get("y", 0.0)
    theta = data.get("theta", 0.0)
    frame_id = data.get("frame_id", "map")
    
    quat_w = math.cos(theta / 2)
    quat_z = math.sin(theta / 2)
    
    goal_msg = (
        "{pose: {header: {frame_id: '" + str(frame_id) + "'}, "
        "pose: {position: {x: " + str(x) + ", y: " + str(y) + ", z: 0.0}, "
        "orientation: {x: 0.0, y: 0.0, z: " + str(quat_z) + ", w: " + str(quat_w) + "}}}}"
    )
    
    result = subprocess.run(
        ["ros2", "action", "send_goal", "/navigate_to_pose", 
         "nav2_msgs/action/NavigateToPose", goal_msg],
        capture_output=True,
        text=True,
        timeout=75,
        check=False
    )
    
    output = result.stdout + result.stderr
    if result.returncode == 0:
        if "Goal finished" in output or "Goal succeeded" in output or "Goal accepted" in output:
            return {"status": "ok", "data": f"Navigation goal sent successfully. Robot navigating to ({x}, {y}, {theta})."}
        else:
            return {"status": "ok", "data": f"Navigation goal sent to ({x}, {y}, {theta}). Status: {output[-300:]}"}
    return {"status": "error", "data": f"Failed to send navigation goal: {result.stderr or result.stdout}"}

async def _try_odom_pose():
    try:
        result = subprocess.run(
            ["ros2", "topic", "echo", "/odom", "--once"],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        if result.returncode == 0 and result.stdout:
            output = result.stdout
            x_match = re.search(r'position:\s*\n\s+x:\s+([-\d.]+)', output)
            y_match = re.search(r'position:\s*\n\s+x:\s+[-\d.]+\s*\n\s+y:\s+([-\d.]+)', output)
            z_match = re.search(r'orientation:\s*\n\s+x:\s+[-\d.]+\s*\n\s+y:\s+[-\d.]+\s*\n\s+z:\s+([-\d.]+)', output)
            w_match = re.search(r'orientation:\s*\n\s+x:\s+[-\d.]+\s*\n\s+y:\s+[-\d.]+\s*\n\s+z:\s+[-\d.]+\s*\n\s+w:\s+([-\d.]+)', output)
            
            if x_match and y_match and z_match and w_match:
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                z = float(z_match.group(1))
                w = float(w_match.group(1))
                theta = 2 * math.atan2(z, w)
                
                return {
                    "status": "ok",
                    "data": {"x": x, "y": y, "theta": theta, "frame_id": "odom"}
                }
    except:
        pass
    
    return {
        "status": "error",
        "data": {"x": 0.0, "y": 0.0, "theta": 0.0, "frame_id": "unknown", "error": "Could not retrieve robot pose"}
    }

async def handle_get_robot_pose(_data):
    try:
        result = subprocess.run(
            ["ros2", "topic", "echo", "/amcl_pose", "--once"],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        if result.returncode == 0 and result.stdout:
            output = result.stdout
            x_match = re.search(r'position:\s*\n\s+x:\s+([-\d.]+)', output)
            y_match = re.search(r'position:\s*\n\s+x:\s+[-\d.]+\s*\n\s+y:\s+([-\d.]+)', output)
            z_match = re.search(r'orientation:\s*\n\s+x:\s+[-\d.]+\s*\n\s+y:\s+[-\d.]+\s*\n\s+z:\s+([-\d.]+)', output)
            w_match = re.search(r'orientation:\s*\n\s+x:\s+[-\d.]+\s*\n\s+y:\s+[-\d.]+\s*\n\s+z:\s+[-\d.]+\s*\n\s+w:\s+([-\d.]+)', output)
            frame_match = re.search(r'frame_id:\s+(\w+)', output)
            
            if x_match and y_match and z_match and w_match:
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                z = float(z_match.group(1))
                w = float(w_match.group(1))
                theta = 2 * math.atan2(z, w)
                frame_id = frame_match.group(1) if frame_match else "map"
                
                return {"status": "ok", "data": {"x": x, "y": y, "theta": theta, "frame_id": frame_id}}
    except:
        pass
    
    return await _try_odom_pose()

async def handle_get_map_info(_data):
    return {
        "status": "ok",
        "data": {
            "map_name": "sim_house_map",
            "semantic_locations": semantic_locations,
            "description": "A simulated house maze environment with 3 detection platforms containing objects: banana, backpack, book, water bottle, cake, and ball"
        }
    }

async def handle_find_semantic_location(data):
    semantic_name = data.get("semantic_name", "").lower()
    
    for name, location in semantic_locations.items():
        if semantic_name in name or name in semantic_name:
            return {
                "status": "ok",
                "data": {
                    "x": location["x"],
                    "y": location["y"],
                    "theta": location["theta"],
                    "description": location.get("description", "")
                }
            }
    
    return {
        "status": "error",
        "data": f"Semantic location '{semantic_name}' not found. Available: {list(semantic_locations.keys())}"
    }

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

    print("Subscribing to ros.topic.info...")
    await bus.subscribe("ros.topic.info", handle_topic_info)
    print("Subscribing to ros.topic.publishers...")
    await bus.subscribe("ros.topic.publishers", handle_topic_publishers)
    print("Subscribing to ros.topic.subscribers...")
    await bus.subscribe("ros.topic.subscribers", handle_topic_subscribers)
    print("Subscribing to ros.topic.subscribe...")
    await bus.subscribe("ros.topic.subscribe", handle_subscribe)
    print("Subscribing to ros.navigate_to_pose...")
    await bus.subscribe("ros.navigate_to_pose", handle_navigate_to_pose)
    print("Subscribing to ros.get_robot_pose...")
    await bus.subscribe("ros.get_robot_pose", handle_get_robot_pose)
    print("Subscribing to ros.get_map_info...")
    await bus.subscribe("ros.get_map_info", handle_get_map_info)
    print("Subscribing to ros.find_semantic_location...")
    await bus.subscribe("ros.find_semantic_location", handle_find_semantic_location)

    print("Enhanced ROS-NATS Adapter Running... (Press Ctrl+C to stop)")
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(run_adapter())
    except KeyboardInterrupt:
        print("\nAdapter stopped.")

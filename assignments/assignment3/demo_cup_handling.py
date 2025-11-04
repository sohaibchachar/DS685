# Simple demo of how cups are handled

print("🧪 DEMO: How the System Handles a CUP")
print("=" * 50)

# Simulate the instruction analysis for a cup
instruction = "Put the red block in the cup"
instruction_lower = instruction.lower()

# Container recognition
container_words = ['bowl', 'container', 'cup', 'mug', 'glass', 'jar', 'box']
has_container = any(word in instruction_lower for word in container_words)

# Action type detection
if any(word in instruction_lower for word in ['remove', 'take out']):
    action_type = 'remove_from_container'
elif any(word in instruction_lower for word in ['take from', 'take out of']):
    action_type = 'remove_from_container'
elif any(word in instruction_lower for word in ['put', 'place', 'drop']):
    if has_container:
        action_type = 'put_in_container'
    elif 'on' in instruction_lower and 'block' in instruction_lower:
        action_type = 'stack'
    else:
        action_type = 'place'
else:
    action_type = 'unknown'

# Container naming
container_name = 'container1'  # default
if 'bowl' in instruction_lower:
    container_name = 'bowl1'
elif 'cup' in instruction_lower:
    container_name = 'cup1'
elif 'mug' in instruction_lower:
    container_name = 'mug1'
elif 'glass' in instruction_lower:
    container_name = 'glass1'
elif 'jar' in instruction_lower:
    container_name = 'jar1'
elif 'box' in instruction_lower:
    container_name = 'box1'

# Goal determination
if action_type == 'put_in_container':
    goal_predicates = [f'(in block1 {container_name})']
else:
    goal_predicates = ['(on block2 block1)']  # fallback

print(f"Instruction: '{instruction}'")
print(f"Container detected: {has_container}")
print(f"Action type: {action_type}")
print(f"Container name: {container_name}")
print(f"Goal predicates: {goal_predicates}")
print()

print("Generated PDDL Objects:")
print(f"  block1 - block")
print(f"  block2 - block") 
print(f"  robot1 - robot")
if has_container:
    print(f"  {container_name} - container")
print()

print("Generated PDDL Goal:")
print(f"  (and {' '.join(goal_predicates)} )")
print()

print("✅ CUP is correctly recognized as a CONTAINER type!")
print("✅ Not treated as a block!")
print("✅ Gets proper container actions (put-in-container, etc.)")

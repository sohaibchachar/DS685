# Assignment 3: Simple Explanation

## 🎯 What is this assignment about?

**In Simple Terms:**

You need to look at videos/pictures of robots doing tasks (like picking up blocks, stacking them, etc.) and automatically create a **planning language file** that describes:
1. **WHAT objects exist** (blocks, robots, tables)
2. **WHAT the robot can DO** (pick up, place, stack)
3. **WHAT different scenarios look like** (problem setups)

Think of it like this:
- **Input**: Videos/images of robots with written instructions like "stack the red block on the blue block"
- **Output**: A special format (PDDL) that a computer can use to plan robot actions

## 📚 Key Concepts Explained

### 1. **DROID Dataset**
- A big collection of robot videos/images
- Shows robots doing real tasks in homes/labs
- Each video has a written instruction describing what the robot is doing
- Example: "Pick up the apple and put it in the box"

### 2. **PDDL (Planning Domain Definition Language)**
- A special file format that describes:
  - **Domain**: What can happen (actions like "pick", "place")
  - **Problem**: A specific scenario (start: blocks on table, goal: stack them)
- Used by AI planners to figure out step-by-step actions

### 3. **Visual Language Models (VLMs)**
- AI models that understand both images AND text
- Can look at a robot video/image and describe what's happening
- Examples: CLIP, BLIP, GPT-4V

## 🔄 The Process (Step by Step)

```
Step 1: Get robot videos from DROID dataset
   ↓
Step 2: Filter for BLOCK-related tasks (your focus)
   ↓
Step 3: Look at instructions like "stack blocks"
   ↓
Step 4: Extract what you learn:
   - Objects: blocks, robot, table
   - Actions: stack, pick, place
   - Relationships: blocks can be on top of each other
   ↓
Step 5: Write PDDL files automatically:
   - domain.pddl = defines actions (pick, place, stack)
   - problem1.pddl = a specific scenario to solve
   - problem2.pddl = another scenario
   - problem3.pddl = yet another scenario
```

## 📋 What You Need to Create

### 1. **domain.pddl** - The "Rule Book"
Defines what can happen in the robot world:

```pddl
(define (domain robot-manipulation)
  (:action pick
    :parameters (robot, block)
    :precondition (robot is empty, block is on table)
    :effect (robot is holding block)
  )
  
  (:action stack
    :parameters (robot, block1, block2)
    :precondition (robot holding block1, block2 is clear)
    :effect (block1 on top of block2)
  )
)
```

### 2. **problem1.pddl, problem2.pddl, problem3.pddl** - Specific Scenarios

Each describes:
- **Initial state**: What the world looks like at the start
  - Example: "Block A is on table, Block B is on table"
- **Goal**: What you want to achieve
  - Example: "Block A should be on top of Block B"

## 🛠️ How to Do It

### Option 1: Automated (What the code does)
1. Read DROID dataset
2. Filter for "block" related instructions
3. Extract keywords: "stack", "pick", "block", etc.
4. Automatically write PDDL files

### Option 2: Manual (You write them yourself)
1. Look at DROID examples
2. Understand what actions are common
3. Write PDDL files by hand

## 🎯 Your Specific Task: BLOCKS WORLD

You're focusing on **block manipulation tasks**:
- Stacking blocks
- Picking up blocks
- Arranging blocks
- Building towers

So you filter the DROID dataset for episodes that mention:
- "block", "stack", "tower", "cube", etc.

## ✅ What Success Looks Like

You'll have:
1. ✅ `domain.pddl` - Describes block manipulation actions
2. ✅ `problem1.pddl` - A block stacking scenario
3. ✅ `problem2.pddl` - Another block scenario  
4. ✅ `problem3.pddl` - Another block scenario
5. ✅ `pddl_generator.py` - Code that creates these files

## 🔍 Example: From Robot Video to PDDL

**Input (DROID dataset):**
- Video: Robot picks up red block, stacks it on blue block
- Instruction: "Stack the red block on the blue block"

**Output (PDDL):**

**domain.pddl:**
```pddl
(:action stack
  :parameters (?robot - robot ?block1 - block ?block2 - block)
  :precondition (holding ?robot ?block1) (clear ?block2)
  :effect (on ?block1 ?block2) (not (holding ?robot ?block1))
)
```

**problem1.pddl:**
```pddl
(:init
  (at red-block table)
  (at blue-block table)
  (clear red-block)
  (clear blue-block)
)
(:goal
  (on red-block blue-block)
)
```

## 🚀 Quick Start

```bash
# 1. Set up Google Cloud authentication (for DROID dataset)
gcloud auth application-default login

# 2. Run the generator (focusing on blocks)
python pddl_generator.py --blocks --episodes 20

# 3. You'll get:
#    - domain.pddl
#    - problem1.pddl  
#    - problem2.pddl
#    - problem3.pddl
```

## 💡 Think of It This Way

**PDDL = Recipe Book for Robots**

- **Domain** = The list of cooking actions (chop, sauté, bake)
- **Problem** = A specific recipe (start: raw ingredients, goal: cooked meal)

Your job: Look at robot cooking videos (DROID dataset), figure out what "cooking actions" robots use, and write it in the robot recipe book format (PDDL).

## 📖 Still Confused?

Ask yourself:
1. What does the robot DO in the videos? → **Actions** (pick, place, stack)
2. What OBJECTS are involved? → **Types** (blocks, robot, table)
3. What CONDITIONS matter? → **Predicates** (holding, on, clear)
4. What are different SCENARIOS? → **Problems** (different starting/ending states)

This assignment is about **automating** the creation of this recipe book by learning from real robot demonstration videos!


#!/usr/bin/env python3
"""
Generate PDDL domain and problem files from video descriptions.
Modified to read from a single video_descriptions.txt file instead of JSON.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env from workspace root (three levels up: assignment3 -> assignments -> eng-ai-agents)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        print(f"⚠️  .env file not found at {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables directly")

# Import OpenAI library
try:
    import openai
except ImportError:
    raise ImportError("openai library not available. Install with: pip install openai")

# --- Configuration ---
DESCRIPTIONS_FILE = "video_descriptions.txt"
OUTPUT_DOMAIN_FILE = "domain.pddl"
OUTPUT_PROBLEM_DIR = "problems"

# OpenAI Configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # Options: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"

# --- PDDL Generation Prompts ---
PDDL_DOMAIN_GENERATION_PROMPT = """You are an expert in PDDL (Planning Domain Definition Language) planning.
Given multiple robot manipulation video descriptions, generate a comprehensive PDDL domain file that covers ALL possible actions and scenarios.

I will provide you with multiple video descriptions. Analyze them all to understand the complete set of:
- Types of objects (blocks, containers, robots)
- Possible predicates (relationships between objects)
- Actions the robot can perform

VIDEO DESCRIPTIONS:
{all_descriptions}

DOMAIN REQUIREMENTS:
Based on the video descriptions above, generate a PDDL domain that includes:

1. Types: block, container, robot

2. Predicates (include all needed based on the descriptions):
   * (on ?x - block ?y - block) - block x is stacked on block y
   * (on-table ?x - block) - block x is on the table
   * (in ?x - block ?c - container) - block x is inside container c
   * (clear ?x - object) - object x has nothing on top of it
   * (holding ?r - robot ?x - block) - robot r is holding block x
   * (empty ?r - robot) - robot r is not holding anything

3. Actions (include all needed based on the descriptions):
   * pick-up: robot picks up a block from table
   * put-down: robot puts down a block on table
   * stack: robot stacks a block on another block
   * unstack: robot unstacks a block from another block
   * put-in: robot puts a block inside a container
   * take-out: robot takes a block out of a container

Ensure the domain is comprehensive enough to solve ALL the problems described in the videos.

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY valid PDDL syntax
- DO NOT use markdown code blocks (no ```lisp, ```pddl, or ``` markers)
- DO NOT add any explanations, comments, or text after the PDDL code
- DO NOT add sections like "### Explanation" or similar
- Output MUST start with "(define (domain" and end with ")" - nothing else

Generate the domain file:

(define (domain robot-manipulation)
  ...
)
"""

PDDL_PROBLEM_GENERATION_PROMPT = """You are an expert in PDDL (Planning Domain Definition Language) planning. 
Given a description of a robot manipulation video, generate a PDDL problem file.

The video description will include:
- Initial state: objects present, their colors, and locations
- Actions performed: what the robot does
- Final state: where objects end up

Your task is to generate a PDDL problem file (define problem, objects, initial state, goal state).

PROBLEM REQUIREMENTS:
- Objects: List all blocks, containers, and robots mentioned
- Initial state: All predicates true at the start
- Goal state: Only include objects that were manipulated in the video or changed location.
              Do NOT include predicates like "clear yellow_block" or "empty robot1" in goal.
              If robot moved yellow_block from table to on top of red_block, only include (on yellow_block red_block) in goal state.

Example:
(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot block1 - block block2 - block block3 - block
  )
  
  (:init
    (empty robot1) (on-table block1) (clear block1) (on-table block2) (clear block2) (on-table block3) (clear block3)
  )
  
  (:goal
    (and (on block2 block1))
  )
)

IMPORTANT:
- Use consistent object names (e.g., green_block, blue_cup, robot1)
- Normalize colors: green, yellow, blue, red, orange, black, white
- Blocks can be stacked on blocks (on predicate)
- Blocks can be in containers (in predicate)
- Only include objects mentioned in the description
- Be precise about initial and goal states

VIDEO DESCRIPTION:
{description}

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY valid PDDL syntax
- DO NOT use markdown code blocks (no ```lisp, ```pddl, or ``` markers)
- DO NOT add any explanations, comments, or text after the PDDL code
- DO NOT add sections like "### Explanation" or "This PDDL..."
- Output MUST start with "(define (problem" and end with ")" - nothing else

Generate the problem file:

(define (problem ...)
  ...
)
"""

# --- Parse Descriptions File ---
def parse_descriptions_file(filepath: Path) -> Dict[str, str]:
    """
    Parse the video_descriptions.txt file into a dictionary.
    Format:
        ================================================================================
        Episode ID: <episode_id>
        Video Path: <path>
        ================================================================================
        <description>
        
    Returns:
        Dictionary mapping episode_id to description
    """
    descriptions = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the separator pattern
    entries = re.split(r'={80,}', content)
    
    # Process entries
    i = 0
    while i < len(entries):
        entry = entries[i].strip()
        
        if not entry:
            i += 1
            continue
        
        # Look for Episode ID
        episode_match = re.search(r'Episode ID:\s*(.+)', entry)
        if episode_match:
            episode_id = episode_match.group(1).strip()
            
            # The description should be in the next entry
            if i + 1 < len(entries):
                description = entries[i + 1].strip()
                if description:
                    descriptions[episode_id] = description
                i += 2
                continue
        
        i += 1
    
    return descriptions

# --- OpenAI Generation Function ---
def generate_with_openai(prompt: str) -> str:
    """Generates text using OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in PDDL planning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4096
    )
    
    return response.choices[0].message.content

# --- PDDL Extraction ---
def extract_pddl_block(response: str, block_type: str) -> Optional[str]:
    """
    Extracts PDDL block (domain or problem) from LLM response.
    Since we instruct the LLM not to add markdown/explanations in the prompt,
    we just need to extract the (define ...) block.
    """
    # Try to find the define block with proper parenthesis matching
    define_start = response.find(f'(define ({block_type}')
    
    if define_start == -1:
        # Try alternative spacing
        define_start = response.find(f'(define({block_type}')
    
    if define_start == -1:
        return None
    
    # Count parentheses to find the matching closing one
    paren_count = 0
    end_pos = define_start
    
    for i in range(define_start, len(response)):
        if response[i] == '(':
            paren_count += 1
        elif response[i] == ')':
            paren_count -= 1
            if paren_count == 0:
                end_pos = i + 1
                break
    
    if end_pos > define_start:
        return response[define_start:end_pos].strip()
    
    return None

def extract_domain_from_response(response: str) -> Optional[str]:
    """Extracts domain PDDL from LLM response."""
    return extract_pddl_block(response, 'domain')

def extract_problem_from_response(response: str) -> Optional[str]:
    """Extracts problem PDDL from LLM response."""
    return extract_pddl_block(response, 'problem')

# --- Main Processing ---
def generate_domain_from_all_descriptions(all_descriptions: Dict[str, str]) -> Optional[str]:
    """Generates PDDL domain from all video descriptions using OpenAI."""
    # Combine all descriptions into a single text
    descriptions_text = ""
    for episode_id, description in all_descriptions.items():
        if not description.startswith("ERROR"):
            descriptions_text += f"\n--- Episode: {episode_id} ---\n{description}\n"
    
    prompt = PDDL_DOMAIN_GENERATION_PROMPT.format(all_descriptions=descriptions_text)
    
    print(f"   🤖 Generating domain with OpenAI ({OPENAI_MODEL})...")
    
    try:
        response = generate_with_openai(prompt)
        domain = extract_domain_from_response(response)
        
        if domain:
            print(f"   ✅ Successfully extracted domain")
            return domain
        else:
            print(f"   ⚠️  Could not extract domain from response")
            print(f"   Response preview: {response[:500]}...")
            return None
            
    except Exception as e:
        print(f"   ⚠️  Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_problem_for_video(description: str) -> Optional[str]:
    """Generates PDDL problem for a single video description using OpenAI."""
    prompt = PDDL_PROBLEM_GENERATION_PROMPT.format(description=description)
    
    print(f"   🤖 Generating problem with OpenAI ({OPENAI_MODEL})...")
    
    try:
        response = generate_with_openai(prompt)
        problem = extract_problem_from_response(response)
        
        if problem:
            print(f"   ✅ Successfully extracted problem")
            return problem
        else:
            print(f"   ⚠️  Could not extract problem from response")
            print(f"   Response preview: {response[:500]}...")
            return None
            
    except Exception as e:
        print(f"   ⚠️  Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def make_safe_episode_id(episode_id: str) -> str:
    """
    Convert episode ID to safe format for filenames.
    Example: RAD+c6cf6b42+2023-08-31-14h-00m-49s -> RAD_c6cf6b42_2023_08_31_14h_00m_49s
    """
    return episode_id.replace('+', '_').replace('-', '_')

def main():
    """Main function to generate PDDL files from descriptions."""
    print("=" * 80)
    print("STAGE 2: Generate PDDL from Video Descriptions")
    print("=" * 80)
    
    # Load descriptions
    script_dir = Path(__file__).parent
    descriptions_file = script_dir / DESCRIPTIONS_FILE
    
    if not descriptions_file.exists():
        print(f"❌ Descriptions file not found: {descriptions_file}")
        print(f"   Run process_all_videos.py first!")
        return
    
    print(f"\n📁 Loading descriptions from: {descriptions_file}")
    all_descriptions = parse_descriptions_file(descriptions_file)
    
    print(f"✅ Loaded {len(all_descriptions)} video descriptions")
    
    # Setup output directories
    output_problem_dir = script_dir / OUTPUT_PROBLEM_DIR
    output_problem_dir.mkdir(parents=True, exist_ok=True)
    output_domain_file = script_dir / OUTPUT_DOMAIN_FILE
    
    # Check OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY=your-key-here")
        print("   Or add it to your .env file")
        return
    
    print(f"\n🔧 Using OpenAI model: {OPENAI_MODEL}")
    
    # Process each video
    successful = 0
    failed = 0
    
    # Generate domain from ALL descriptions
    print(f"\n📝 Generating domain file from all video descriptions...")
    domain = generate_domain_from_all_descriptions(all_descriptions)
    
    if domain:
        with open(output_domain_file, 'w', encoding='utf-8') as f:
            f.write(domain)
        print(f"✅ Saved domain: {output_domain_file}")
    else:
        print(f"⚠️  Could not generate domain file")
        print(f"   Cannot proceed without domain. Exiting.")
        return
    
    # Generate problem for each video
    print(f"\n📝 Generating problem files...")
    for i, (episode_id, description) in enumerate(all_descriptions.items(), 1):
        print(f"\n[{i}/{len(all_descriptions)}] {episode_id}")
        
        # Skip if description contains errors
        if description.startswith("ERROR"):
            print(f"   ⚠️  Skipping due to error in description")
            failed += 1
            continue
        
        problem = generate_problem_for_video(description)
        
        if problem:
            safe_episode_id = make_safe_episode_id(episode_id)
            problem_file = output_problem_dir / f"{safe_episode_id}.pddl"
            
            with open(problem_file, 'w', encoding='utf-8') as f:
                f.write(problem)
            
            print(f"   ✅ Saved: {problem_file.name}")
            successful += 1
        else:
            failed += 1
    
    print(f"\n✅ Completed!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"📁 Domain: {output_domain_file}")
    print(f"📁 Problems: {output_problem_dir}")

if __name__ == "__main__":
    main()



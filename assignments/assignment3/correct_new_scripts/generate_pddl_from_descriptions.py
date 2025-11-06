#!/usr/bin/env python3
"""
Stage 2: Generate PDDL Domain and Problem files from video descriptions using an LLM.
This script takes the text descriptions from Stage 1 and uses an LLM to generate PDDL files.
Supports multiple LLM backends: Local models (Phi-3, Llama), OpenAI API, Anthropic API.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env from workspace root (two levels up from this script)
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
except ImportError:
    pass  # dotenv not available, will use environment variables directly

# Try importing LLM libraries
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# --- Configuration ---
DESCRIPTIONS_FILE = "correct_new_scripts/video_descriptions/all_descriptions.json"
OUTPUT_DOMAIN_FILE = "correct_new_scripts/domain.pddl"
OUTPUT_PROBLEM_DIR = "correct_new_scripts/problems"

# LLM Configuration
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai")  # Default to OpenAI. Options: "local", "openai", "anthropic"
LOCAL_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # Fallback: open, no auth required
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # Options: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # or "claude-3-opus-20240229"

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

# --- LLM Loading (Local) ---
def load_local_llm():
    """Loads a local LLM model using transformers."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library not available. Install with: pip install transformers")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for local LLM. Use API backend instead.")
    
    print(f"🔧 Loading local LLM: {LOCAL_MODEL_ID}")
    device = "cuda"
    
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"  # Use eager attention to avoid cache compatibility issues
        )
        
        model.eval()
        print(f"   ✅ Loaded: {LOCAL_MODEL_ID}")
        return {"tokenizer": tokenizer, "model": model, "device": device}
    except Exception as e:
        print(f"   ⚠️  Failed to load: {e}")
        raise

# --- LLM Generation Functions ---
def generate_with_local_llm(llm_dict: Dict, prompt: str) -> str:
    """Generates text using a local LLM."""
    tokenizer = llm_dict["tokenizer"]
    model = llm_dict["model"]
    device = llm_dict["device"]
    
    # Format prompt
    user_prompt = prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            pass  # Fallback to plain prompt
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Determine pad_token_id
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                use_cache=False,  # Disable cache for Phi-3 compatibility
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )
        except Exception as gen_error:
            # Fallback: try without cache and with different parameters
            print(f"      ⚠️  Generation error: {gen_error}")
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    output_attentions=False,
                    output_hidden_states=False
                )
            except Exception as e2:
                print(f"      ⚠️  Generation failed again: {e2}")
                raise e2
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Remove the prompt from response
    if user_prompt in response:
        response = response.split(user_prompt)[-1].strip()
    elif prompt in response:
        response = response[len(prompt):].strip()
    
    return response

def generate_with_openai(prompt: str) -> str:
    """Generates text using OpenAI API."""
    if not HAS_OPENAI:
        raise RuntimeError("openai library not available. Install with: pip install openai")
    
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

def generate_with_anthropic(prompt: str) -> str:
    """Generates text using Anthropic API."""
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic library not available. Install with: pip install anthropic")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

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
def generate_domain_from_all_descriptions(all_descriptions_dict: Dict, llm_backend: str, llm_dict: Optional[Dict] = None) -> Optional[str]:
    """Generates PDDL domain from all video descriptions."""
    # Combine all descriptions into a single text
    descriptions_text = ""
    for episode_id, desc_data in all_descriptions_dict.items():
        description = desc_data.get("description", "")
        if not description.startswith("ERROR"):
            descriptions_text += f"\n--- Episode: {episode_id} ---\n{description}\n"
    
    prompt = PDDL_DOMAIN_GENERATION_PROMPT.format(all_descriptions=descriptions_text)
    
    print(f"   🤖 Generating domain with {llm_backend}...")
    
    try:
        if llm_backend == "local":
            response = generate_with_local_llm(llm_dict, prompt)
        elif llm_backend == "openai":
            response = generate_with_openai(prompt)
        elif llm_backend == "anthropic":
            response = generate_with_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")
        
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

def generate_problem_for_video(description: str, llm_backend: str, llm_dict: Optional[Dict] = None) -> Optional[str]:
    """Generates PDDL problem for a single video description."""
    prompt = PDDL_PROBLEM_GENERATION_PROMPT.format(description=description)
    
    print(f"   🤖 Generating problem with {llm_backend}...")
    
    try:
        if llm_backend == "local":
            response = generate_with_local_llm(llm_dict, prompt)
        elif llm_backend == "openai":
            response = generate_with_openai(prompt)
        elif llm_backend == "anthropic":
            response = generate_with_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")
        
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

def main():
    """Main function to generate PDDL files from descriptions."""
    print("=" * 80)
    print("STAGE 2: Generate PDDL from Video Descriptions")
    print("=" * 80)
    
    # Load descriptions
    descriptions_file = Path(DESCRIPTIONS_FILE)
    if not descriptions_file.exists():
        print(f"❌ Descriptions file not found: {descriptions_file}")
        print(f"   Run Stage 1 (extract_video_descriptions.py) first!")
        return
    
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        all_descriptions = json.load(f)
    
    print(f"\n📁 Loaded {len(all_descriptions)} video descriptions")
    
    # Setup output directories
    output_problem_dir = Path(OUTPUT_PROBLEM_DIR)
    output_problem_dir.mkdir(parents=True, exist_ok=True)
    output_domain_file = Path(OUTPUT_DOMAIN_FILE)
    
    # Load LLM
    llm_backend = LLM_BACKEND.lower()
    llm_dict = None
    
    print(f"\n🔧 Using LLM backend: {llm_backend}")
    
    if llm_backend == "local":
        try:
            llm_dict = load_local_llm()
        except Exception as e:
            print(f"⚠️  Local LLM failed: {e}")
            print("   Set LLM_BACKEND=openai or LLM_BACKEND=anthropic to use API")
            return
    elif llm_backend == "openai":
        if not HAS_OPENAI:
            print("❌ OpenAI library not available. Install with: pip install openai")
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            print("   Set it with: export OPENAI_API_KEY=your-key-here")
            print("   Or add it to your .env file")
            return
        print(f"   ✅ Using OpenAI model: {OPENAI_MODEL}")
    elif llm_backend == "anthropic":
        if not HAS_ANTHROPIC:
            print("❌ Anthropic library not available. Install with: pip install anthropic")
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("❌ ANTHROPIC_API_KEY environment variable not set")
            return
    else:
        print(f"❌ Unknown LLM backend: {llm_backend}")
        print("   Options: local, openai, anthropic")
        return
    
    # Process each video
    successful = 0
    failed = 0
    
    # Generate domain from ALL descriptions
    print(f"\n📝 Generating domain file from all video descriptions...")
    domain = generate_domain_from_all_descriptions(all_descriptions, llm_backend, llm_dict)
    
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
    for i, (episode_id, desc_data) in enumerate(all_descriptions.items(), 1):
        print(f"\n[{i}/{len(all_descriptions)}] {episode_id}")
        
        description = desc_data["description"]
        
        # Skip if description contains errors
        if description.startswith("ERROR"):
            print(f"   ⚠️  Skipping due to error in description")
            failed += 1
            continue
        
        problem = generate_problem_for_video(description, llm_backend, llm_dict)
        
        if problem:
            safe_episode_id = episode_id.replace('+', '_').replace('-', '_')
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

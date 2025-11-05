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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # Options: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # or "claude-3-opus-20240229"

# --- PDDL Generation Prompt ---
PDDL_GENERATION_PROMPT = """You are an expert in PDDL (Planning Domain Definition Language) planning. 
Given a description of a robot manipulation video, generate PDDL domain and problem files.

The video description will include:
- Initial state: objects present, their colors, and locations
- Actions performed: what the robot does
- Final state: where objects end up

Your task is to generate:
1. A PDDL domain file (define domain, types, predicates, actions)
2. A PDDL problem file (define problem, objects, initial state, goal state)

DOMAIN REQUIREMENTS:
- Types: block, container, robot
- Predicates:
  * (on ?x - block ?y - block) - block x is stacked on block y
  * (on-table ?x - block) - block x is on the table
  * (in ?x - block ?c - container) - block x is inside container c
  * (clear ?x - object) - object x has nothing on top of it
  * (holding ?r - robot ?x - block) - robot r is holding block x
  * (empty ?r - robot) - robot r is not holding anything

- Actions:
  * pick-up: robot picks up a block from table
  * put-down: robot puts down a block on table
  * stack: robot stacks a block on another block
  * unstack: robot unstack a block from another block
  * put-in: robot puts a block inside a container
  * take-out: robot takes a block out of a container

PROBLEM REQUIREMENTS:
- Objects: List all blocks, containers, and robots mentioned
- Initial state: All predicates true at the start
- Goal state: All predicates that should be true at the end

IMPORTANT:
- Use consistent object names (e.g., green_block, blue_cup, robot1)
- Normalize colors: green, yellow, blue, red, orange, black, white
- Blocks can be stacked on blocks (on predicate)
- Blocks can be in containers (in predicate)
- Only include objects mentioned in the description (blocks and containers only)
- Be precise about initial and goal states

Given the video description below, generate the domain and problem PDDL files.

VIDEO DESCRIPTION:
{description}

Generate both domain and problem files. Format your response as:

=== DOMAIN ===
(define (domain ...)
  ...
)

=== PROBLEM ===
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
def extract_pddl_from_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts domain and problem PDDL from LLM response."""
    # Try to extract domain and problem sections
    domain_match = re.search(r'=== DOMAIN ===\s*(.*?)(?=== PROBLEM ===|$)', response, re.DOTALL)
    problem_match = re.search(r'=== PROBLEM ===\s*(.*?)$', response, re.DOTALL)
    
    if domain_match and problem_match:
        domain = domain_match.group(1).strip()
        problem = problem_match.group(1).strip()
        
        # Clean up domain (remove markdown code blocks if present)
        domain = re.sub(r'```pddl?\s*', '', domain)
        domain = re.sub(r'```\s*$', '', domain, flags=re.MULTILINE)
        
        # Clean up problem (remove markdown code blocks if present)
        problem = re.sub(r'```pddl?\s*', '', problem)
        problem = re.sub(r'```\s*$', '', problem, flags=re.MULTILINE)
        
        return domain, problem
    
    # Fallback: try to find PDDL blocks
    domain_match = re.search(r'\(define\s+\(domain[^)]+\)[^)]*\)', response, re.DOTALL)
    problem_match = re.search(r'\(define\s+\(problem[^)]+\)[^)]*\)', response, re.DOTALL)
    
    if domain_match and problem_match:
        return domain_match.group(0), problem_match.group(0)
    
    return None, None

# --- Main Processing ---
def generate_pddl_for_video(description: str, llm_backend: str, llm_dict: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Generates PDDL domain and problem for a single video description."""
    prompt = PDDL_GENERATION_PROMPT.format(description=description)
    
    print(f"   🤖 Generating PDDL with {llm_backend}...")
    
    try:
        if llm_backend == "local":
            response = generate_with_local_llm(llm_dict, prompt)
        elif llm_backend == "openai":
            response = generate_with_openai(prompt)
        elif llm_backend == "anthropic":
            response = generate_with_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")
        
        domain, problem = extract_pddl_from_response(response)
        
        if domain and problem:
            print(f"   ✅ Successfully extracted domain and problem")
            return domain, problem
        else:
            print(f"   ⚠️  Could not extract PDDL from response")
            print(f"   Response preview: {response[:500]}...")
            return None, None
            
    except Exception as e:
        print(f"   ⚠️  Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
    
    # Generate domain once (use first video or combine all)
    print(f"\n📝 Generating domain file...")
    first_description = list(all_descriptions.values())[0]["description"]
    domain, _ = generate_pddl_for_video(first_description, llm_backend, llm_dict)
    
    if domain:
        with open(output_domain_file, 'w', encoding='utf-8') as f:
            f.write(domain)
        print(f"✅ Saved domain: {output_domain_file}")
    else:
        print(f"⚠️  Could not generate domain file")
    
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
        
        _, problem = generate_pddl_for_video(description, llm_backend, llm_dict)
        
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

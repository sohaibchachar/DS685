#!/usr/bin/env python3
"""Test script to verify goal predicate filtering works correctly."""

import sys
from pathlib import Path

# Add the current directory to path so we can import the function
sys.path.insert(0, str(Path(__file__).parent))

from generate_pddl_with_cosmos import generate_problem_pddl
import tempfile

def test_goal_filtering():
    """Test that goal predicates only include manipulated objects."""
    
    # Test case 1: Block in container (should only have 'in' predicate, not 'on-table' for container)
    print("Test Case 1: Block put in container")
    print("-" * 70)
    
    test_analysis_1 = {
        "initial_state": {
            "objects": [
                {"name": "green block", "color": "green", "location": "on table"},
                {"name": "black bowl", "color": "black", "location": "on table"}
            ]
        },
        "final_state": {
            "objects": [
                {"name": "green block", "color": "green", "location": "in black bowl"},
                {"name": "black bowl", "color": "black", "location": "on table"}
            ]
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generate_problem_pddl(
            "test_episode_1",
            "Put the green block in the black bowl",
            test_analysis_1,
            output_dir
        )
        
        problem_file = output_dir / "problem_test_episode_1.pddl"
        if problem_file.exists():
            content = problem_file.read_text()
            print("Generated problem file:")
            print(content)
            print()
            
            # Check that goal only has 'in' predicate, not 'on-table black_bowl'
            if "(in green_block black_bowl)" in content:
                print("✅ PASS: Goal contains manipulation predicate '(in green_block black_bowl)'")
            else:
                print("❌ FAIL: Goal missing manipulation predicate")
            
            # Check that goal does NOT have redundant 'on-table black_bowl'
            goal_section = content.split("(:goal")[1].split(")")[0] if "(:goal" in content else ""
            if "(on-table black_bowl)" not in goal_section:
                print("✅ PASS: Goal correctly excludes redundant '(on-table black_bowl)' predicate")
            else:
                print("❌ FAIL: Goal incorrectly includes redundant '(on-table black_bowl)' predicate")
        else:
            print("❌ FAIL: Problem file not generated")
    
    print("\n" + "=" * 70)
    
    # Test case 2: Block stacked on another block
    print("Test Case 2: Block stacked on another block")
    print("-" * 70)
    
    test_analysis_2 = {
        "initial_state": {
            "objects": [
                {"name": "red block", "color": "red", "location": "on table"},
                {"name": "blue block", "color": "blue", "location": "on table"}
            ]
        },
        "final_state": {
            "objects": [
                {"name": "red block", "color": "red", "location": "on table"},
                {"name": "blue block", "color": "blue", "location": "on red block"}
            ]
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generate_problem_pddl(
            "test_episode_2",
            "Stack the blue block on the red block",
            test_analysis_2,
            output_dir
        )
        
        problem_file = output_dir / "problem_test_episode_2.pddl"
        if problem_file.exists():
            content = problem_file.read_text()
            print("Generated problem file:")
            print(content)
            print()
            
            # Check that goal only has 'on' predicate, not 'on-table red_block'
            goal_section = content.split("(:goal")[1].split(")")[0] if "(:goal" in content else ""
            if "(on blue_block red_block)" in goal_section:
                print("✅ PASS: Goal contains manipulation predicate '(on blue_block red_block)'")
            else:
                print("❌ FAIL: Goal missing manipulation predicate")
            
            if "(on-table red_block)" not in goal_section:
                print("✅ PASS: Goal correctly excludes redundant '(on-table red_block)' predicate")
            else:
                print("❌ FAIL: Goal incorrectly includes redundant '(on-table red_block)' predicate")
        else:
            print("❌ FAIL: Problem file not generated")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")

if __name__ == "__main__":
    test_goal_filtering()


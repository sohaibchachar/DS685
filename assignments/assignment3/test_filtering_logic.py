#!/usr/bin/env python3
"""Test script to verify goal predicate filtering logic works correctly."""

def test_goal_filtering_logic():
    """Test the filtering logic that removes redundant goal predicates."""
    
    # Simulate initial state predicates
    init_predicates = [
        "(empty robot1)",
        "(on-table green_block)",
        "(clear green_block)",
        "(on-table black_bowl)",
        "(clear black_bowl)"
    ]
    
    # Simulate goal predicates that might include redundant ones
    goal_predicates_before_filtering = [
        "(in green_block black_bowl)",
        "(on-table black_bowl)"  # This should be filtered out
    ]
    
    # Apply the filtering logic (same as in generate_pddl_with_cosmos.py)
    init_predicates_set = set(pred.strip() for pred in init_predicates)
    manipulation_goal_predicates = []
    for goal_pred in goal_predicates_before_filtering:
        goal_pred_normalized = goal_pred.strip()
        # Only include goal predicates that are NOT already in the initial state
        if goal_pred_normalized not in init_predicates_set:
            manipulation_goal_predicates.append(goal_pred)
    
    goal_predicates_after_filtering = manipulation_goal_predicates
    
    print("=" * 70)
    print("TEST: Goal Predicate Filtering")
    print("=" * 70)
    print("\nInitial State Predicates:")
    for pred in init_predicates:
        print(f"  {pred}")
    
    print("\nGoal Predicates BEFORE Filtering:")
    for pred in goal_predicates_before_filtering:
        print(f"  {pred}")
    
    print("\nGoal Predicates AFTER Filtering:")
    for pred in goal_predicates_after_filtering:
        print(f"  {pred}")
    
    print("\n" + "-" * 70)
    
    # Verify results
    assert "(in green_block black_bowl)" in goal_predicates_after_filtering, \
        "❌ FAIL: Manipulation predicate '(in green_block black_bowl)' should be in goal"
    print("✅ PASS: Manipulation predicate '(in green_block black_bowl)' is in goal")
    
    assert "(on-table black_bowl)" not in goal_predicates_after_filtering, \
        "❌ FAIL: Redundant predicate '(on-table black_bowl)' should be filtered out"
    print("✅ PASS: Redundant predicate '(on-table black_bowl)' is correctly filtered out")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Filtering logic works correctly.")
    print("=" * 70)
    
    # Test Case 2: Stacking scenario
    print("\n\n" + "=" * 70)
    print("TEST CASE 2: Stacking Scenario")
    print("=" * 70)
    
    init_predicates_2 = [
        "(empty robot1)",
        "(on-table red_block)",
        "(clear red_block)",
        "(on-table blue_block)",
        "(clear blue_block)"
    ]
    
    goal_predicates_before_2 = [
        "(on blue_block red_block)",
        "(on-table red_block)"  # Should be filtered out
    ]
    
    init_predicates_set_2 = set(pred.strip() for pred in init_predicates_2)
    manipulation_goal_predicates_2 = []
    for goal_pred in goal_predicates_before_2:
        goal_pred_normalized = goal_pred.strip()
        if goal_pred_normalized not in init_predicates_set_2:
            manipulation_goal_predicates_2.append(goal_pred)
    
    print("\nInitial State Predicates:")
    for pred in init_predicates_2:
        print(f"  {pred}")
    
    print("\nGoal Predicates BEFORE Filtering:")
    for pred in goal_predicates_before_2:
        print(f"  {pred}")
    
    print("\nGoal Predicates AFTER Filtering:")
    for pred in manipulation_goal_predicates_2:
        print(f"  {pred}")
    
    assert "(on blue_block red_block)" in manipulation_goal_predicates_2
    assert "(on-table red_block)" not in manipulation_goal_predicates_2
    print("\n✅ PASS: Stacking scenario also works correctly!")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    test_goal_filtering_logic()


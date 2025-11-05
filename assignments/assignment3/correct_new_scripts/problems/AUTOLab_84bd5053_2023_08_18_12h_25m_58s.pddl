(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  (:objects
    yellow_block green_block red_block - block
    blue_cup - container
    robot1 - robot
  )
  (:init
    (on-table yellow_block)
    (on-table green_block)
    (on-table red_block)
    (on-table blue_cup)
    (clear yellow_block)
    (clear green_block)
    (clear red_block)
    (clear blue_cup)
    (empty robot1)
  )
  (:goal
    (and
      (in yellow_block blue_cup)
      (in green_block blue_cup)
      (in red_block blue_cup)
      (on-table blue_cup)
    )
  )
)

This PDDL setup defines the actions and states based on the initial and final states described in the video. The domain includes all necessary predicates and actions, while the problem specifies the objects, initial conditions, and the desired goal state.
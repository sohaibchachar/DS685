(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  (:objects 
    yellow_block green_block red_triangle - block
    blue_cup - container
    robot1 - robot
  )
  (:init 
    (on-table yellow_block)
    (on-table green_block)
    (on-table red_triangle)
    (on-table blue_cup)
    (clear yellow_block)
    (clear green_block)
    (clear red_triangle)
    (clear blue_cup)
    (empty robot1)
  )
  (:goal 
    (in yellow_block blue_cup)
  )
)

This representation captures the initial and goal states as well as the actions performed by the robot, including the necessary preconditions and effects for each action.
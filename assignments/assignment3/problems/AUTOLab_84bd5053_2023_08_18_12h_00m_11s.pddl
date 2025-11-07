(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot
    yellow_block - block
    green_block - block
    blue_cup - container
    red_triangle - block
  )
  
  (:init
    (empty robot1)
    (on-table yellow_block)
    (on-table green_block)
    (on-table blue_cup)
    (on-table red_triangle)
    (clear yellow_block)
    (clear green_block)
    (clear red_triangle)
    (clear blue_cup)
  )
  
  (:goal
    (and (in yellow_block blue_cup))
  )
)
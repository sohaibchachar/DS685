(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects 
    green_block - block
    white_cup - container
    robot_arm - robot
  )
  
  (:init 
    (on-table green_block)
    (on-table white_cup)
    (clear green_block)
    (clear white_cup)
    (empty robot_arm)
  )
  
  (:goal 
    (and 
      (in green_block white_cup)
      (on-table white_cup)
      (clear white_cup)
    )
  )
)

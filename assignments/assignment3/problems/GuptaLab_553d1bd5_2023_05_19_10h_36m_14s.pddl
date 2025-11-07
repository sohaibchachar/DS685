(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects
    robot_arm - robot
    red_block - block
    green_block - block
    white_cabinet - container
  )
  
  (:init
    (on-table red_block)
    (on-table green_block)
    (above white_cabinet table)
    (above robot_arm table)
  )
  
  (:goal
    (and (on red_block green_block))
  )
)
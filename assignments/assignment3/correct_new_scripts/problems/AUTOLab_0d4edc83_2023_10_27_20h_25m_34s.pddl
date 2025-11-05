(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects 
    green_cube - block
    black_pan - container
    robot_arm - robot
  )
  
  (:init 
    (on-table green_cube)
    (on-table black_pan)
    (clear green_cube)
    (clear black_pan)
    (empty robot_arm)
  )
  
  (:goal 
    (in green_cube black_pan)
  )
)
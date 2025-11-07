(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot
    white_bowl - container
    green_cube - block
    blue_cube - block
    yellow_cube - block
  )
  
  (:init
    (on-table white_bowl)
    (in blue_cube white_bowl)
    (in green_cube white_bowl)
    (on-table yellow_cube)
    (empty robot1)
  )
  
  (:goal
    (and
      (on-table green_cube)
    )
  )
)
(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects
    green_cylindrical_object red_block - block
    blue_bowl yellow_bowl - container
    robot_arm - robot
  )

  (:init
    (on-table green_cylindrical_object)
    (on-table red_block)
    (on-table blue_bowl)
    (on-table yellow_bowl)
    (clear green_cylindrical_object)
    (clear red_block)
    (clear blue_bowl)
    (clear yellow_bowl)
    (clear robot_arm)
    (empty robot_arm)
  )

  (:goal
    (and
      (in green_cylindrical_object blue_bowl)
      (on red_block green_cylindrical_object)
      (clear blue_bowl)
    )
  )
)

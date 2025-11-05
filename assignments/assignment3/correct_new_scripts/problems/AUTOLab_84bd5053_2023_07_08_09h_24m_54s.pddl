(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects 
    yellow_block orange_block green_block white_block - block
    cylindrical_object - container
    robot1 - robot
  )
  
  (:init 
    (on-table yellow_block)
    (on-table orange_block)
    (on-table green_block)
    (on-table white_block)
    (clear yellow_block)
    (clear orange_block)
    (clear green_block)
    (clear white_block)
    (clear cylindrical_object)
    (empty robot1)
    (not (in yellow_block cylindrical_object))
    (not (in orange_block cylindrical_object))
    (not (in green_block cylindrical_object))
  )
  
  (:goal 
    (and 
      (in yellow_block cylindrical_object)
      (in orange_block cylindrical_object)
      (in green_block cylindrical_object)
      (clear cylindrical_object)
    )
  )
)

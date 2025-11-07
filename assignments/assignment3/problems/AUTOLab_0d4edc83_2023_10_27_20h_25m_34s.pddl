(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot green_block - block black_bowl - container
  )
  
  (:init
    (empty robot1) (on-table green_block) (clear green_block) (on-table black_bowl) (clear black_bowl)
  )
  
  (:goal
    (and (in green_block black_bowl))
  )
)
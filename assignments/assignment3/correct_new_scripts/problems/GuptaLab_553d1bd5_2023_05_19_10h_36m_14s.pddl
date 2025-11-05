```lisp
(define (problem block-manipulation-problem)
  (:domain block-manipulation)
  
  (:objects
    red_block green_block - block
    robot1 - robot
  )

  (:init
    (on-table red_block)
    (on-table green_block)
    (clear red_block)
    (clear green_block)
    (empty robot1)
  )

  (:goal
    (and
      (on red_block green_block)
      (clear green_block)
    )
  )
)

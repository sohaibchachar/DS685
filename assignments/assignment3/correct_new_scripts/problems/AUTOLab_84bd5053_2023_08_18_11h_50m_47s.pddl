```lisp
(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)

  (:objects
    yellow_block red_block - block
    blue_cup - container
    robot1 - robot
  )

  (:init
    (on-table yellow_block)
    (on-table red_block)
    (on-table blue_cup)
    (clear yellow_block)
    (clear red_block)
    (clear blue_cup)
    (empty robot1)
  )

  (:goal
    (and
      (in yellow_block blue_cup)
      (on-table red_block)
      (on-table blue_cup)
      (clear red_block)
    )
  )
)

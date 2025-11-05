```lisp
(define (problem robot-manipulation-problem)
  (:domain robot-manipulation)
  (:objects
    white_bowl - container
    green_cube - block
    straw - block
    blue_cup - container
    robot1 - robot
  )
  (:init
    (on-table white_bowl)
    (in green_cube white_bowl)
    (on-table straw)
    (on-table blue_cup)
    (clear white_bowl)
    (clear straw)
    (clear blue_cup)
    (empty robot1)
  )
  (:goal
    (and
      (on-table green_cube)
      (on-table white_bowl)
      (on-table straw)
      (on-table blue_cup)
      (clear green_cube)
    )
  )
)

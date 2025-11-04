(define (problem RAIL_d027f2ae_2023_06_05_16h_33m_01s)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot block1 - block block2 - block block3 - block
  )
  
  (:init
    (empty robot1) (on-table block1) (clear block1) (on-table block2) (clear block2) (on-table block3) (clear block3)
  )
  
  (:goal
    (and (on block2 block1))
  )
)

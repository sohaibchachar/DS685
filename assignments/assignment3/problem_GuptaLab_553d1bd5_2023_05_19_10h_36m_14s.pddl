(define (problem GuptaLab_553d1bd5_2023_05_19_10h_36m_14s)
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

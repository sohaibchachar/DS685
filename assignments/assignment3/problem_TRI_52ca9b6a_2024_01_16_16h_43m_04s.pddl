(define (problem TRI_52ca9b6a_2024_01_16_16h_43m_04s)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot block1 - block block2 - block
  )
  
  (:init
    (empty robot1) (on-table block1) (clear block1) (on-table block2) (clear block2)
  )
  
  (:goal
    (and (on block2 block1))
  )
)

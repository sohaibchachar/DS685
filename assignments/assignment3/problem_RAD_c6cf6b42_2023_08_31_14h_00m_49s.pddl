(define (problem RAD_c6cf6b42_2023_08_31_14h_00m_49s)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot block1 - block bowl1 - container
  )
  
  (:init
    (empty robot1) (in block1 bowl1) (clear bowl1)
  )
  
  (:goal
    (and (on-table block1))
  )
)

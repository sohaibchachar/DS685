(define (problem AUTOLab_84bd5053_2023_08_18_11h_50m_47s)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot block1 - block cup1 - container
  )
  
  (:init
    (empty robot1) (on-table block1) (clear block1) (clear cup1)
  )
  
  (:goal
    (and (in block1 cup1))
  )
)

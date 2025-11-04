(define (problem AUTOLab_0d4edc83_2023_10_27_20h_25m_34s)
  (:domain robot-manipulation)
  
  (:objects
    robot1 - robot block1 - block bowl1 - container
  )
  
  (:init
    (empty robot1) (on-table block1) (clear block1) (clear bowl1)
  )
  
  (:goal
    (and (in block1 bowl1))
  )
)

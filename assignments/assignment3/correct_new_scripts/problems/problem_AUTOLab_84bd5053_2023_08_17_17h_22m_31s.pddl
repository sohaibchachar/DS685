(define (problem AUTOLab_84bd5053_2023_08_17_17h_22m_31s)
    (:domain robot-manipulation)
    (:objects
        blue_cup - container
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear blue_cup)
        (clear yellow_block)
        (empty robot1)
        (on-table blue_cup)
        (on-table yellow_block)
    )
    (:goal
        (in yellow_block blue_cup)
    )
)

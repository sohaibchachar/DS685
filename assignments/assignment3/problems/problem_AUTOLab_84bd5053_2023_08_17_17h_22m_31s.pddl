(define (problem AUTOLab_84bd5053_2023_08_17_17h_22m_31s)
    (:domain robot-manipulation)
    (:objects
        yellow_block - block
        blue_cup - container
        robot1 - robot
    )
    (:init
        (empty robot1)
        (on-table yellow_block)
        (clear yellow_block)
        (on-table blue_cup)
        (clear blue_cup)
    )
    (:goal
        (and
            (in yellow_block blue_cup)
            (on-table blue_cup)
        )
    )
)

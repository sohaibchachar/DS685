(define (problem AUTOLab_84bd5053_2023_08_18_12h_00m_11s)
    (:domain robot-manipulation)
    (:objects
        yellow_block - block
        blue_cup - container
        green_block - block
        robot1 - robot
    )
    (:init
        (empty robot1)
        (on-table yellow_block)
        (clear yellow_block)
        (on-table blue_cup)
        (clear blue_cup)
        (on-table green_block)
        (clear green_block)
    )
    (:goal
        (and
            (in yellow_block blue_cup)
            (on-table blue_cup)
            (on-table green_block)
        )
    )
)

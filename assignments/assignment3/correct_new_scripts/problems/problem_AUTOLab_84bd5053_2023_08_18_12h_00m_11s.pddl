(define (problem AUTOLab_84bd5053_2023_08_18_12h_00m_11s)
    (:domain robot-manipulation)
    (:objects
        blue_cup - container
        green_block - block
        red_block - block
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear blue_cup)
        (clear green_block)
        (clear red_block)
        (clear yellow_block)
        (empty robot1)
        (on-table blue_cup)
        (on-table green_block)
        (on-table red_block)
        (on-table yellow_block)
    )
    (:goal
        (in yellow_block blue_cup)
    )
)

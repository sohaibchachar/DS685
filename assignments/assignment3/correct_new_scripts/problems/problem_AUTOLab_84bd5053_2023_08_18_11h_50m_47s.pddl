(define (problem AUTOLab_84bd5053_2023_08_18_11h_50m_47s)
    (:domain robot-manipulation)
    (:objects
        blue_cup - container
        red_block - block
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear blue_cup)
        (clear red_block)
        (clear yellow_block)
        (empty robot1)
        (on-table blue_cup)
        (on-table red_block)
        (on-table yellow_block)
    )
    (:goal
        (in red_block blue_cup)
    )
)

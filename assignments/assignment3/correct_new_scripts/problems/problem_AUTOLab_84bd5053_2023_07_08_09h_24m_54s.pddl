(define (problem AUTOLab_84bd5053_2023_07_08_09h_24m_54s)
    (:domain robot-manipulation)
    (:objects
        blue_block - block
        green_block - block
        grey_cup - container
        red_block - block
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear blue_block)
        (clear green_block)
        (clear grey_cup)
        (clear red_block)
        (clear yellow_block)
        (empty robot1)
        (on-table blue_block)
        (on-table green_block)
        (on-table grey_cup)
        (on-table red_block)
        (on-table yellow_block)
    )
    (:goal
        (and
            (in green_block grey_cup)
            (in red_block grey_cup)
            (in yellow_block grey_cup)
        )
    )
)

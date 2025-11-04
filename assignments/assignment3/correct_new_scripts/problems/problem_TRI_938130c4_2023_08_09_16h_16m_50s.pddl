(define (problem TRI_938130c4_2023_08_09_16h_16m_50s)
    (:domain robot-manipulation)
    (:objects
        blue_cup - container
        green_block - block
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear blue_cup)
        (clear green_block)
        (clear yellow_block)
        (empty robot1)
        (on-table blue_cup)
        (on-table green_block)
        (on-table yellow_block)
    )
    (:goal
        (in green_block blue_cup)
    )
)

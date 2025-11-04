(define (problem TRI_52ca9b6a_2024_01_16_16h_43m_04s)
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
        (on red_block green_block)
    )
)

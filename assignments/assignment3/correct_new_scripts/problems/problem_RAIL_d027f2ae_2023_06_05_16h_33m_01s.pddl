(define (problem RAIL_d027f2ae_2023_06_05_16h_33m_01s)
    (:domain robot-manipulation)
    (:objects
        blue_block - block
        green_block - block
        orange_block - block
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear blue_block)
        (clear green_block)
        (clear orange_block)
        (clear yellow_block)
        (empty robot1)
        (on-table blue_block)
        (on-table green_block)
        (on-table orange_block)
        (on-table yellow_block)
    )
    (:goal
        (on orange_block green_block)
    )
)

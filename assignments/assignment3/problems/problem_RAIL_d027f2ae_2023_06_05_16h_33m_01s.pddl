(define (problem RAIL_d027f2ae_2023_06_05_16h_33m_01s)
    (:domain robot-manipulation)
    (:objects
        green_block - block
        orange_block - block
        yellow_block - block
        blue_block - block
        robot1 - robot
    )
    (:init
        (empty robot1)
        (on-table green_block)
        (clear green_block)
        (on-table orange_block)
        (clear orange_block)
        (on-table yellow_block)
        (clear yellow_block)
        (on-table blue_block)
        (clear blue_block)
    )
    (:goal
        (and
            (on orange_block green_block)
            (on-table yellow_block)
            (on-table blue_block)
        )
    )
)

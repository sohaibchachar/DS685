(define (problem GuptaLab_553d1bd5_2023_05_19_10h_36m_14s)
    (:domain robot-manipulation)
    (:objects
        red_block - block
        green_block - block
        robot1 - robot
    )
    (:init
        (empty robot1)
        (on-table red_block)
        (clear red_block)
        (on-table green_block)
        (clear green_block)
    )
    (:goal
        (on-table red_block)
    )
)

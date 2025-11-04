(define (problem GuptaLab_553d1bd5_2023_05_19_10h_36m_14s)
    (:domain robot-manipulation)
    (:objects
        green_block - block
        red_block - block
        robot1 - robot
    )
    (:init
        (clear green_block)
        (clear red_block)
        (empty robot1)
        (on-table green_block)
        (on-table red_block)
    )
    (:goal
        (on red_block green_block)
    )
)

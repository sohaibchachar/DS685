(define (problem AUTOLab_0d4edc83_2023_10_27_20h_25m_34s)
    (:domain robot-manipulation)
    (:objects
        black_bowl - container
        green_block - block
        robot1 - robot
        yellow_block - block
    )
    (:init
        (clear black_bowl)
        (clear green_block)
        (clear yellow_block)
        (empty robot1)
        (on-table black_bowl)
        (on-table green_block)
        (on-table yellow_block)
    )
    (:goal
        (in green_block black_bowl)
    )
)

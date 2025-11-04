(define (problem test_AUTOLab_0d4edc83_2023_10_27_20h_25m_34s)
    (:domain robot-manipulation)
    (:objects
        green_block - block
        black_bowl - container
        robot1 - robot
    )
    (:init
        (empty robot1)
        (on-table green_block)
        (clear green_block)
        (on-table black_bowl)
        (clear black_bowl)
    )
    (:goal
        (in green_block black_bowl)
    )
)

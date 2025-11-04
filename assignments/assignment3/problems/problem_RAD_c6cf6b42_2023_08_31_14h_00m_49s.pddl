(define (problem RAD_c6cf6b42_2023_08_31_14h_00m_49s)
    (:domain robot-manipulation)
    (:objects
        green_block - block
        white_bowl - container
        blue_cup - container
        robot1 - robot
    )
    (:init
        (empty robot1)
        (in green_block white_bowl)
        (on-table white_bowl)
        (clear white_bowl)
        (on-table blue_cup)
        (clear blue_cup)
    )
    (:goal
        (and
            (on-table green_block)
            (on-table white_bowl)
            (on-table blue_cup)
        )
    )
)

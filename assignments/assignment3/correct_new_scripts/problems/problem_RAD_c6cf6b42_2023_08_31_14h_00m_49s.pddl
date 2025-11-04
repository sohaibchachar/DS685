(define (problem RAD_c6cf6b42_2023_08_31_14h_00m_49s)
    (:domain robot-manipulation)
    (:objects
        green_block - block
        robot1 - robot
        white_bowl - container
        white_container - container
    )
    (:init
        (clear green_block)
        (clear white_bowl)
        (clear white_container)
        (empty robot1)
        (on-table green_block)
        (on-table white_bowl)
        (on-table white_container)
    )
    (:goal
        (in green_block white_bowl)
    )
)

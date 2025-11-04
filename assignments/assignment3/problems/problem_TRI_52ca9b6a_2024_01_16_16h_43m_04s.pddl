(define (problem TRI_52ca9b6a_2024_01_16_16h_43m_04s)
    (:domain robot-manipulation)
    (:objects
        green_block - block
        red_block - block
        yellow_block - block
        blue_cup - container
        white_bowl - container
        clear_cup - container
        robot1 - robot
    )
    (:init
        (empty robot1)
        (on-table green_block)
        (clear green_block)
        (on-table red_block)
        (clear red_block)
        (on-table yellow_block)
        (clear yellow_block)
        (on-table blue_cup)
        (clear blue_cup)
        (on-table white_bowl)
        (clear white_bowl)
        (on-table clear_cup)
        (clear clear_cup)
    )
    (:goal
        (and
            (on-table green_block)
            (on red_block green_block)
            (on yellow_block green_block)
            (on-table blue_cup)
            (on-table white_bowl)
            (on-table clear_cup)
        )
    )
)

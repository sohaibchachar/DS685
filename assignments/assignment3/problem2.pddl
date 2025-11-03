(define (problem problem2)
    (:domain robot-manipulation)
    
    (:objects
        robot1 - robot
        box - object item - object
    )
    
    (:init
        (clear box) (clear item) (handempty robot1) (at robot1 item)
    )
    
    (:goal
        (and (on item box) (clear box))
    )
)

(define (problem problem3)
    (:domain robot-manipulation)
    
    (:objects
        robot1 - robot
        cup - object table - object
    )
    
    (:init
        (clear cup) (clear table) (handempty robot1) (at robot1 cup)
    )
    
    (:goal
        (and (on cup table) (placed cup table))
    )
)

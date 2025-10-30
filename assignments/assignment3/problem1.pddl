(define (problem problem1)
    (:domain robot-manipulation)
    
    (:objects
        robot1 - robot
        object1 - object object2 - object target - object
    )
    
    (:init
        (clear object1) (clear object2) (handempty robot1) (on object1 target) (on object2 target)
    )
    
    (:goal
        (holding robot1 object1)
    )
)

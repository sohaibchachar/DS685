(define (domain robot-manipulation)
    
    (:requirements :strips :typing)
    
    (:types
        object - physobj
        robot
        gripper
    )
    
    (:predicates
        (at ?r - robot ?x - object)
        (holding ?r - robot ?x - object)
        (on ?x - object ?y - object)
        (clear ?x - object)
        (handempty ?r - robot)
        (picked ?x - object)
        (placed ?x - object ?y - object)
    )
    
    (:action pick
        :parameters (?r - robot ?x - object)
        :precondition (and (at ?r ?x) (clear ?x) (handempty ?r))
        :effect (and (holding ?r ?x) (not (handempty ?r)) (not (at ?r ?x)) (picked ?x))
    )
    
    (:action place
        :parameters (?r - robot ?x - object ?y - object)
        :precondition (and (holding ?r ?x))
        :effect (and (not (holding ?r ?x)) (handempty ?r) (at ?r ?x) (on ?x ?y) (placed ?x ?y))
    )
    
    (:action place-floor
        :parameters (?r - robot ?x - object)
        :precondition (and (holding ?r ?x))
        :effect (and (not (holding ?r ?x)) (handempty ?r) (at ?r ?x) (clear ?x))
    )
)

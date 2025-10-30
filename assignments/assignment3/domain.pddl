(define (domain robot-manipulation)
    (:requirements :strips :typing)
    (:types
        object
        robot
    )
    (:predicates
        (at ?r - robot ?o - object)
        (clear ?o - object)
        (holding ?r - robot ?o - object)
        (on ?o1 - object ?o2 - object)
        (open ?o - object)
        (closed ?o - object)
    )
    (:action pick
        :parameters (?r - robot ?o - object)
        :precondition (and (clear ?o))
        :effect (and (holding ?r ?o) (not (clear ?o)))
    )
    (:action place
        :parameters (?r - robot ?o - object ?dst - object)
        :precondition (and (holding ?r ?o))
        :effect (and (on ?o ?dst) (not (holding ?r ?o)) (clear ?o))
    )
)

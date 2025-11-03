(define (domain robot-manipulation)
    (:requirements :strips :typing)
    (:types
        block - object
        container - object
        surface - object
        robot - agent
    )
    (:predicates
        (holding ?r - robot ?o - block)
        (on ?o1 - block ?o2 - object)
        (clear ?o - block)
        (in ?o - block ?c - container)
        (on-table ?o - block)
        (empty ?c - container)
        (open ?c - container)
        (closed ?c - container)
    )
    (:action pick
        :parameters (?r - robot ?o - block)
        :precondition (and (clear ?o) (on-table ?o))
        :effect (and (holding ?r ?o) (not (clear ?o)) (not (on-table ?o)))
    )
    (:action place
        :parameters (?r - robot ?o - block ?dst - surface)
        :precondition (and (holding ?r ?o))
        :effect (and (on-table ?o) (clear ?o) (not (holding ?r ?o)))
    )
    (:action put-in
        :parameters (?r - robot ?o - block ?c - container)
        :precondition (and (holding ?r ?o) (open ?c))
        :effect (and (in ?o ?c) (not (holding ?r ?o)) (clear ?o))
    )
    

)

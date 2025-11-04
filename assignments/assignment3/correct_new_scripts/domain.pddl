(define (domain robot-manipulation)
    (:requirements :strips :typing)
    
    (:types
        block - object
        container - object
        robot - object
    )
    
    (:predicates
        (on ?x - block ?y - block)
        (on-table ?x - block)
        (in ?x - block ?c - container)
        (clear ?x - object)
        (holding ?r - robot ?x - block)
        (empty ?r - robot)
    )
    
    (:action pick-up
        :parameters (?r - robot ?x - block)
        :precondition (and (empty ?r) (on-table ?x) (clear ?x))
        :effect (and (holding ?r ?x) (not (empty ?r)) (not (on-table ?x)) (not (clear ?x)))
    )

    (:action put-down
        :parameters (?r - robot ?x - block)
        :precondition (holding ?r ?x)
        :effect (and (empty ?r) (on-table ?x) (clear ?x) (not (holding ?r ?x)))
    )

    (:action stack
        :parameters (?r - robot ?x - block ?y - block)
        :precondition (and (holding ?r ?x) (clear ?y))
        :effect (and (on ?x ?y) (clear ?x) (empty ?r) (not (holding ?r ?x)) (not (clear ?y)))
    )

    (:action unstack
        :parameters (?r - robot ?x - block ?y - block)
        :precondition (and (empty ?r) (on ?x ?y) (clear ?x))
        :effect (and (holding ?r ?x) (clear ?y) (not (empty ?r)) (not (on ?x ?y)) (not (clear ?x)))
    )

    (:action put-in-container
        :parameters (?r - robot ?x - block ?c - container)
        :precondition (and (holding ?r ?x) (clear ?c))
        :effect (and (in ?x ?c) (empty ?r) (not (holding ?r ?x)) (not (clear ?c)))
    )

    (:action take-from-container
        :parameters (?r - robot ?x - block ?c - container)
        :precondition (and (empty ?r) (in ?x ?c))
        :effect (and (holding ?r ?x) (clear ?c) (not (empty ?r)) (not (in ?x ?c)))
    )
)

(define (domain AUTOLab_84bd5053_2023_08_18_12h_00m_11s)
    (:requirements :strips :typing)
    (:types
        block container robot surface - object
    )
    (:predicates
        (holding ?r - robot ?b - block)
        (on ?o1 - block ?o2 - object)
        (clear ?o - block)
        (in ?o - block ?c - container)
        (on-table ?o - block)
        (empty ?c - container)
        (open ?c - container)
        (closed ?c - container)
    )

    (:action put_in_container
        :parameters (?r - robot ?b - block ?c - container)
        :precondition (and (on-table ?b) (clear ?b))
        :effect (and (holding ?r ?b))
    )
)

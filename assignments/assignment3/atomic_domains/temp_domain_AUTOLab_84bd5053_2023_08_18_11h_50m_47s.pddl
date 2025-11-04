(define (domain AUTOLab_84bd5053_2023_08_18_11h_50m_47s)
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

    (:action stack_block
        :parameters (?r - robot ?b - block ?c - container)
        :precondition (and (on-table ?b) (clear ?b))
        :effect (and (holding ?r ?b))
    )
)

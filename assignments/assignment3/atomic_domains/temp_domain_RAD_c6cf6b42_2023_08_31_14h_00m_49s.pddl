(define (domain RAD_c6cf6b42_2023_08_31_14h_00m_49s)
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

    (:action manipulate_object
        :parameters (?r - robot ?b - block ?c - container)
        :precondition (and (on-table ?b) (clear ?b))
        :effect (and (holding ?r ?b))
    )
)

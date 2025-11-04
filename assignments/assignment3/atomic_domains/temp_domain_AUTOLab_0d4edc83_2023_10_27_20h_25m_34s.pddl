(define (domain AUTOLab_0d4edc83_2023_10_27_20h_25m_34s)
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

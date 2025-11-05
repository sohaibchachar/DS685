
(define (domain robot-manipulation)
  (:requirements :strips)
  (:types block container robot)

  (:predicates
    (on ?x - block ?y - block)
    (on-table ?x - block)
    (in ?x - block ?c - container)
    (clear ?x - block)
    (holding ?r - robot ?x - block)
    (empty ?r - robot)
  )

  (:action pick-up
    :parameters (?r - robot ?b - block)
    :precondition (and (on-table ?b) (clear ?b) (empty ?r))
    :effect (and (holding ?r ?b) (not (on-table ?b)) (not (clear ?b)) (not (empty ?r)))
  )

  (:action put-down
    :parameters (?r - robot ?b - block)
    :precondition (holding ?r ?b)
    :effect (and (on-table ?b) (clear ?b) (not (holding ?r ?b)) (not (empty ?r)))
  )

  (:action stack
    :parameters (?r - robot ?b1 - block ?b2 - block)
    :precondition (and (holding ?r ?b1) (clear ?b2))
    :effect (and (on ?b1 ?b2) (not (holding ?r ?b1)) (not (clear ?b2)) (not (clear ?b1)))
  )

  (:action unstack
    :parameters (?r - robot ?b1 - block ?b2 - block)
    :precondition (on ?b1 ?b2)
    :effect (and (holding ?r ?b1) (clear ?b1) (not (on ?b1 ?b2)) (not (clear ?b2)))
  )

  (:action put-in
    :parameters (?r - robot ?b - block ?c - container)
    :precondition (holding ?r ?b)
    :effect (and (in ?b ?c) (not (holding ?r ?b)) (not (clear ?b)))
  )

  (:action take-out
    :parameters (?r - robot ?b - block ?c - container)
    :precondition (in ?b ?c)
    :effect (and (holding ?r ?b) (not (in ?b ?c)) (not (clear ?b)))
  )
)

=
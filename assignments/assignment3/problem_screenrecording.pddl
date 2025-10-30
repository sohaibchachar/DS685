(define (problem problem_screenrecording)
    (:domain robot-manipulation)
    (:objects
        robot1 - robot
        obj1 obj2 table - object
    )
    (:init
        (clear obj1) (clear obj2)
    )
    (:goal
        (on obj1 table)
    )
    ; Top labels (label, prob, frame): [('a robot picking an object', 0.7253717184066772, '0003.jpg'), ('a robot picking an object', 0.6609562635421753, '0002.jpg'), ('moving an object on a table', 0.6019740104675293, '0007.jpg'), ('moving an object on a table', 0.5008769035339355, '0005.jpg'), ('a robot picking an object', 0.4687666893005371, '0006.jpg')]
    ; Prompts used: ['a robot picking an object', 'placing an object on a surface', 'stacking two objects', 'opening a container', 'closing a container', 'moving an object on a table', 'putting an object inside a box']
)

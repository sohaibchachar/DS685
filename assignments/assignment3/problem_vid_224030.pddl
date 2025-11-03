(define (problem problem_vid_224030)
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
    ; Top labels (label, prob, frame): [('a robot picking an object', 0.9491737484931946, '0005.jpg'), ('a robot picking an object', 0.9027298092842102, '0007.jpg'), ('a robot picking an object', 0.7118456959724426, '0004.jpg'), ('a robot picking an object', 0.5570182800292969, '0006.jpg'), ('moving an object on a table', 0.5529594421386719, '0002.jpg')]
    ; Prompts used: ['a robot picking an object', 'placing an object on a surface', 'stacking two objects', 'opening a container', 'closing a container', 'moving an object on a table', 'putting an object in a box']
)

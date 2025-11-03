(define (problem problem_vid_223920)
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
    ; Top labels (label, prob, frame): [('a robot picking an object', 0.728146493434906, '0007.jpg'), ('a robot picking an object', 0.7032734751701355, '0003.jpg'), ('moving an object on a table', 0.665648877620697, '0008.jpg'), ('a robot picking an object', 0.6634323000907898, '0001.jpg'), ('a robot picking an object', 0.5678333044052124, '0002.jpg')]
    ; Prompts used: ['a robot picking an object', 'placing an object on a surface', 'stacking two objects', 'opening a container', 'closing a container', 'moving an object on a table', 'putting an object in a box']
)

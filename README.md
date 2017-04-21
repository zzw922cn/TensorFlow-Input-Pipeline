# TensorFlow-Input-Pipeline
Input Pipeline Examples based on multi-threads and FIFOQueue in TensorFlow, including mini-batching training.

## Result

[-] Run `python big_input.py`, you can get:

```
[array([ 6, 10], dtype=int32), array([[[ 1. ,  2.1,  3.5],
        [ 4. ,  5. ,  6. ]],

       [[ 1. ,  2. ,  3.5],
        [ 4. ,  5. ,  6. ]]])]
[array([8], dtype=int32), array([[[ 1. ,  2. ,  3.5],
        [ 4. ,  5.5,  6. ]]])]
```

[-] Run `python small_input.py`, you can get:

```
[array([[0, 1],
       [6, 7]], dtype=int32), 16]
==============================
[array([[2, 3],
       [4, 5]], dtype=int32), 16]
==============================
[array([[10, 11],
       [12, 13]], dtype=int32), 144]
==============================
[array([[8, 9]], dtype=int32), 81]
==============================
pass
```

# TensorFlow-Input-Pipeline
Input Pipeline Examples based on multi-threads and FIFOQueue in TensorFlow, including mini-batching training.

## Graphs

### Graph for small dataset example code
![image](https://github.com/zzw922cn/TensorFlow-Input-Pipeline/blob/master/img/small_graph.png)

### Graph for big dataset example code
![image](https://github.com/zzw922cn/TensorFlow-Input-Pipeline/blob/master/img/big_graph.png)

## Usage

If your dataset is too large to load once, you can first convert your dataset to TFRecords files, my example code shows how to write data into TFRecords and how to read data from TFRecords correctly. You can run the example code `python big_input.py`, and you'll see following result:

```
[array([ 6, 10], dtype=int32), array([[[ 1. ,  2.1,  3.5],
        [ 4. ,  5. ,  6. ]],

       [[ 1. ,  2. ,  3.5],
        [ 4. ,  5. ,  6. ]]])]
[array([8], dtype=int32), array([[[ 1. ,  2. ,  3.5],
        [ 4. ,  5.5,  6. ]]])]
```

Well, if your dataset is not large enough, you can totally load your dataset once and then use FIFOQueue to improve training. You can also run the example code `python small_input.py`, and you'll see following result:

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

## Contact Me
For any questions, welcome to send email to :**zzw922cn@gmail.com**. If you use wechat, you can follow me by searching wechat public media id:**deeplearningdigest**, I would push several articles every week to share my deep learning practices with you. Thanks!

# TensorFlow-Input-Pipeline
Input Pipeline Examples based on multi-threads and FIFOQueue in TensorFlow, including mini-batching training.

## Graphs

### Graph for small dataset example code
![image](https://github.com/zzw922cn/TensorFlow-Input-Pipeline/blob/master/img/small.png)

### Graph for big dataset example code
![image](https://github.com/zzw922cn/TensorFlow-Input-Pipeline/blob/master/img/big.png)

## Usage

If your dataset is too large to load once, you can first convert your dataset to TFRecords files, my example code shows how to write data into TFRecords and how to read data from TFRecords correctly. You can run the example code `python big_input.py`:

```
usage: small_input.py [-h] [--scale SCALE] [--logdir LOGDIR]
                      [--samples_num SAMPLES_NUM] [--time_length TIME_LENGTH]
                      [--feature_size FEATURE_SIZE] [--num_epochs NUM_EPOCHS]
                      [--batch_size BATCH_SIZE] [--num_classes NUM_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --scale SCALE         specify your dataset scale
  --logdir LOGDIR       specify the location to store log or model
  --samples_num SAMPLES_NUM
                        specify your total number of samples
  --time_length TIME_LENGTH
                        specify max time length of sample
  --feature_size FEATURE_SIZE
                        specify feature size of sample
  --num_epochs NUM_EPOCHS
                        specify number of training epochs
  --batch_size BATCH_SIZE
                        specify batch size when training
  --num_classes NUM_CLASSES
                        specify number of output classes

```

Well, if your dataset is not large enough, you can totally load your dataset once and then use FIFOQueue to improve training. You can also run the example code `python small_input.py`:

```
usage: big_input.py [-h] [--scale SCALE] [--logdir LOGDIR]
                    [--samples_num SAMPLES_NUM] [--time_length TIME_LENGTH]
                    [--feature_size FEATURE_SIZE] [--num_epochs NUM_EPOCHS]
                    [--batch_size BATCH_SIZE] [--num_classes NUM_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --scale SCALE         specify your dataset scale
  --logdir LOGDIR       specify the location to store log or model
  --samples_num SAMPLES_NUM
                        specify your total number of samples
  --time_length TIME_LENGTH
                        specify max time length of sample
  --feature_size FEATURE_SIZE
                        specify feature size of sample
  --num_epochs NUM_EPOCHS
                        specify number of training epochs
  --batch_size BATCH_SIZE
                        specify batch size when training
  --num_classes NUM_CLASSES
                        specify number of output classes

```

## Contact Me
For any questions, welcome to send email to :**zzw922cn@gmail.com**. If you use wechat, you can follow me by searching wechat public media id:**deeplearningdigest**, I would push several articles every week to share my deep learning practices with you. Thanks!

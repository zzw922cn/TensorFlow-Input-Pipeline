#-*- coding:utf-8 -*-
#!/usr/bin/python
''' TensorFlow pipeline for big dataset
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2017-4-15
'''

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.python.training import queue_runner_impl
import os
import numpy as np


# for any data type except int
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# for int data type
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## write tfrecord file
class RecordWriter(object):
  def __init__(self, path):
    self.path = path

  def write(self, content, filename, feature_num=2):
    tfrecords_filename = os.path.join(self.path, filename)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    if feature_num>1:
      assert isinstance(content, list), 'content must be a list now'
      feature_dict = {}
      for i in range(feature_num):
        feature = content[i]
        if isinstance(feature, int):
          feature_dict['feature'+str(i+1)]=_int64_feature(feature)
        else:
          feature_raw = np.array(feature).tostring()
          feature_dict['feature'+str(i+1)]=_bytes_feature(feature_raw)
      features_to_write = tf.train.Example(features=tf.train.Features(feature=feature_dict))
      writer.write(features_to_write.SerializeToString())
      writer.close()
      print('Record has been writen:'+tfrecords_filename)


## read tfrecord file
def read(filename_queue, feature_num=2, dtypes=[list, int]):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  feature_dict={}
  for i in range(feature_num):
    # here, only three data types are allowed: tf.float32, tf.int64, tf.string
    if dtypes[i] is int:
      feature_dict['feature'+str(i+1)]=tf.FixedLenFeature([], tf.int64)
    else:
      feature_dict['feature'+str(i+1)]=tf.FixedLenFeature([], tf.string)
  features = tf.parse_single_example(
      serialized_example,
      features=feature_dict)
  return features

#======================================================================================
## test code
rw = RecordWriter('/home/pony/github/seq2seq_tf/seq2seq_tf/pipeline/')
# you can take a1, a2, a3 as training samples, and take b1, b2, b3 as labels
a1 = np.array([[1,2,3.5],[4,5,6]])
a2 = np.array([[1,2,3.5],[4,5.5,6]])
a3 = np.array([[1,2.1,3.5],[4,5,6]])
b1 = 10
b2 = 8
b3 = 6
rw.write([b1,a1], 'test1.tfrecords')
rw.write([b2,a2], 'test2.tfrecords')
rw.write([b3,a3], 'test3.tfrecords')

fq = tf.train.string_input_producer(['test1.tfrecords', 'test2.tfrecords', 'test3.tfrecords'], 1, shuffle=False)

# read tfrecords, you should specify the data type of each feature
# egs, int, float, list, string
features = read(fq, dtypes=[int, list])
aa = tf.cast(features['feature1'], tf.int32)

bb = tf.decode_raw(features['feature2'], tf.float64)
bb = tf.reshape(bb, [2,3])

aaa, bbb = tf.train.shuffle_batch([aa, bb], batch_size=2, capacity=30, num_threads=1, allow_smaller_final_batch=True, min_after_dequeue=10)

sess = tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
coord = tf.train.Coordinator()
threads = queue_runner_impl.start_queue_runners(sess=sess)

num_batchs = 2
for i in range(num_batchs):
  print sess.run([aaa,bbb])

coord.request_stop()
coord.join(threads)

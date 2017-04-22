#-*- coding:utf-8 -*-
#!/usr/bin/python
''' TensorFlow pipeline for small dataset
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
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import os
import time
import math

flags.DEFINE_string("scale", "small", "specify your dataset scale")
flags.DEFINE_string("logdir", "/home/pony/github/data/inputpipeline", "specify the location to store log or model")
flags.DEFINE_integer("samples_num", 80, "specify your total number of samples")
flags.DEFINE_integer("time_length", 100, "specify max time length of sample")
flags.DEFINE_integer("feature_size", 39, "specify feature size of sample")
flags.DEFINE_integer("num_epochs", 100, "specify number of training epochs")
flags.DEFINE_integer("batch_size", 8, "specify batch size when training")
FLAGS = flags.FLAGS

if __name__ == '__main__':

  scale = FLAGS.scale
  logdir = FLAGS.logdir
  sn = FLAGS.samples_num
  tl = FLAGS.time_length
  fs = FLAGS.feature_size
  num_epochs = FLAGS.num_epochs
  batch_size = FLAGS.batch_size
  num_batches = int(math.ceil(1.0*sn/batch_size))

  with tf.variable_scope('train-samples'):
    x = tf.reshape(tf.range(sn*tl*fs), [sn, tl, fs])
  with tf.variable_scope('train-labels'):
    y = tf.reshape(tf.range(sn*tl), [sn, tl])

  # dequeue ops
  with tf.variable_scope('InputProducer'):
    slice_x, slice_y = tf.train.slice_input_producer([x, y], num_epochs = num_epochs, seed=22, capacity=36, shuffle=True)
    batched_x, batched_y = tf.train.batch([slice_x, slice_y], batch_size=batch_size, dynamic_pad=False, allow_smaller_final_batch=True)
    batched_x = tf.layers.dense(batched_x, 2*fs)
    batched_x = tf.layers.dense(batched_x, fs)
  with tf.variable_scope('Loss'):
    loss = tf.squared_difference(tf.reduce_mean(batched_x), tf.reduce_mean(batched_y))
    tf.summary.scalar('Loss', loss)
  merged = tf.summary.merge_all()

  t1 = time.time()
  sess = tf.Session()
  checkpoint_path = os.path.join(logdir, scale+'_model')
  writer = tf.summary.FileWriter(logdir, sess.graph)
  sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
  coord = tf.train.Coordinator()
  threads = queue_runner_impl.start_queue_runners(sess=sess)
  saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
  saver.save(sess, checkpoint_path)
  for i in range(num_batches*num_epochs):
    l, summary = sess.run([loss, merged])
    writer.add_summary(summary, i)
    print 'Epoch:'+str(i)+'\tLoss:'+str(l)
  writer.close()
  coord.request_stop()
  coord.join(threads)
  print 'program takes time:'+str(time.time()-t1)

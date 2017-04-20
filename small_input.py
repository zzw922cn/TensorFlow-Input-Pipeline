import tensorflow as tf
from tensorflow.python.training import queue_runner_impl

if __name__ == '__main__':

  # if your dataset is small enough to load once
  # the first dimension of a and b must be the same

  sn = 7
  tl = 2
  fs = 2
  a = tf.reshape(tf.range(sn*tl*fs), [sn, tl, fs])
  b = tf.reshape(tf.range(sn*tl), [sn, tl])

  num_epochs = 1
  batch_size = 2
  num_batches = 4

  # dequeue ops
  a_batched, b_batched = tf.train.slice_input_producer([a, b], num_epochs = num_epochs, seed=22, capacity=36, shuffle=True)
  aa, bb = tf.train.batch([a_batched, b_batched], batch_size=batch_size, dynamic_pad=False, allow_smaller_final_batch=True)
  aa3 = tf.reduce_mean(aa)
  bb3 = tf.reduce_mean(bb)
  loss = tf.squared_difference(aa3, bb3)
  sess = tf.Session()
  sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
  coord = tf.train.Coordinator()
  threads = queue_runner_impl.start_queue_runners(sess=sess)
  for i in range(num_batches*num_epochs):
    print sess.run([bb, loss])
    print '='*30
  print 'pass'
  coord.request_stop()
  coord.join(threads)

  

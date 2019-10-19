#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""

搭建CNN模型，实现枸杞分类判别

"""

################## load packages #################
import tensorflow as tf
import numpy as np


########### batch size ############
batch_size=128
epoch = 20
display_step=20
learning_rate=0.01
n_classes=6


########## placeholder ##########
x=tf.placeholder(tf.float32,[None, 212, 212, 95])
y=tf.placeholder(tf.float32,[None, n_classes])


########## label one hot ##########
def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label


##################### read TFRecord and output batch #####################
def read_and_decode(filename, batch_size):
    '''
    filename: TFRecord路径
    '''

    ########### 根据文件名生成一个队列 ############
    filename_queue = tf.train.string_input_producer([filename])

    ########### 生成 TFRecord 读取器 ############
    reader = tf.TFRecordReader()

    ########### 返回文件名和文件 ############
    _, serialized_example = reader.read(filename_queue)

    ########### 取出example里的features #############
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64)})

    ########### 将序列化的img转为uint8的tensor #############
    img = tf.decode_raw(features['img'], tf.float32)

    ########### 将label转为int32的tensor #############
    label = tf.cast(features['label'], tf.int32)

    ########### 将height和width转为int32的tensor #############
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    ########### 将图片调整成正确的尺寸 ###########
    img = tf.reshape(img, [95, 212, 212])

    ########### 批量输出图片, 使用shuffle_batch可以有效地随机从训练数据中抽出batch_size个数据样本 ###########
    ##### shuffle batch之前，必须提前定义影像的size，size不可以是tensor，必须是明确的数字 ######
    ##### num_threads 表示可以选择用几个线程同时读取 #####
    ##### min_after_dequeue 表示读取一次之后队列至少需要剩下的样例数目 #####
    ##### capacity 表示队列的容量 #####
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=100, num_threads=1,
                                                    min_after_dequeue=10, allow_smaller_final_batch=True)

    return img_batch, label_batch


################### cnn model ##################
def model(x, n_classes):

    ####### first conv ########
    #### conv ####
    conv1=tf.layers.conv2d(x, filters=120, kernel_size=7, strides=2, padding='VALID')

    #### BN ####
    conv1=tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1=tf.nn.relu(conv1)

    ####### first conv ########
    #### conv ####
    conv2=tf.layers.conv2d(conv1, filters=240, kernel_size=5, strides=2, padding='VALID')

    #### BN ####
    conv2=tf.layers.batch_normalization(conv2)

    #### relu ####
    conv2=tf.nn.relu(conv2)


    ####### first conv ########
    #### conv ####
    conv3=tf.layers.conv2d(conv2, filters=480, kernel_size=3, strides=2, padding='VALID')

    #### BN ####
    conv3=tf.layers.batch_normalization(conv3)

    #### relu ####
    conv3=tf.nn.relu(conv3)


    ####### first conv ########
    #### conv ####
    conv4=tf.layers.conv2d(conv3, filters=560, kernel_size=3, strides=2, padding='VALID')

    #### BN ####
    conv4=tf.layers.batch_normalization(conv4)

    #### relu ####
    conv4=tf.nn.relu(conv4)


    ####### 全局平均池化 ########
    pool=tf.nn.avg_pool(conv4,ksize=[1,10,10,1],strides=[1,10,10,1],padding='VALID')

    ####### flatten 影像展平 ########
    flatten = tf.reshape(pool, (-1, 1*1*560))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out=tf.layers.dense(flatten, n_classes)

    return out


########## define model, loss and optimizer ##########
#### model pred 影像判断结果 ####
pred=model(x, n_classes)

#### loss 损失计算 ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(tf.nn.softmax(pred),1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


########### tfrecords path ############
filename="gouqi_train_212.tfrecords"

########### get batch img and label ############
img_batch_train, label_batch_train = read_and_decode(filename, batch_size)


################### sess ######################
with tf.Session() as sess:

  ########### 初始化 ###########
  init = tf.global_variables_initializer()
  sess.run(init)

  step=1

  ########## 启动队列线程 ##########
  coord=tf.train.Coordinator()
  threads= tf.train.start_queue_runners(sess=sess, coord=coord)

  ########### train ###########
  batch_idxs = int(360/batch_size)


  for i in range(epoch):
    for j in range(batch_idxs):
      step += 1
      
      ######### 取出img_batch and label_batch #########
     

      X_train, y_train = sess.run([img_batch_train, label_batch_train])
      y_train = one_hot(y_train, 6)

      ##### optimizer ####
      sess.run(optimizer, feed_dict={x: X_train, y: y_train})
      
      ##### show loss and acc ##### 
      if step % display_step==0:
          loss,acc=sess.run([cost, accuracy],feed_dict={x: X_train, y: y_train})
          print("Epoch "+ str(i) + ", Step "+ str(step) + ", Minibatch Loss=" + \
              "{:.6f}".format(loss) + ", Training Accuracy= "+ \
              "{:.5f}".format(acc))


  coord.request_stop()
  coord.join(threads)

sess.close()


print("Optimizer Finished!")



########### tfrecords path ############
filename = "gouqi_test_212.tfrecords"

batch_size=100

########### get batch img and label ############
img_batch_test, label_batch_test = read_and_decode(filename, batch_size)

################### sess ######################
with tf.Session() as sess:

    ########### 初始化 ###########
    init = tf.global_variables_initializer()
    sess.run(init)

    ########## 启动队列线程 ##########
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    X_test, y_test = sess.run([img_batch_test, label_batch_test])

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

    coord.request_stop()
    coord.join(threads)

sess.close()
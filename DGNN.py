#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a lstm netwrok to predict nuclide concentration'


__author__ = 'Zhangyijing'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.app.flags.DEFINE_float('lr', 0.005, 'the learning rate')
tf.app.flags.DEFINE_float('overlapped_percent', 0.5, 'the percentage of overlapped data')
tf.app.flags.DEFINE_float('test_percent', 0.3, 'the percentage of validation data set')
tf.app.flags.DEFINE_boolean('with_overlap', True, 'whether overlapping or not')
tf.app.flags.DEFINE_boolean('test', True, 'whether test the LSTM or not')
tf.app.flags.DEFINE_string('normalization_mode', 'normal', 'normalization method: linear, normal, etc')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the size of each batch')
tf.app.flags.DEFINE_integer('input_size', 3, 'the size of the input')
tf.app.flags.DEFINE_integer('timestep_size', 24, 'the length of the truncated LSTM chain')
tf.app.flags.DEFINE_integer('hidden_dim', 16, 'the dimensions of the state, i.e., the hidden layer')
tf.app.flags.DEFINE_integer('layer_num', 1, 'the number of the stacker LSTM cells')
tf.app.flags.DEFINE_integer('class_num', 4, 'for prediction problem, the class number equals 1')
tf.app.flags.DEFINE_integer('epochs', 500, 'iteration epochs')
tf.app.flags.DEFINE_string('folder_dir', '/Volumes/FAT/项目/EFDC/数据/data/', 'the directory of the data file')
tf.app.flags.DEFINE_string('decay_rate', 'Scheme2-12.3a/', 'decay rate be one of 0, 70d, 250d, 5a, 12.3a')
tf.app.flags.DEFINE_list('locations', [1, 3, 5, 15, 17, 20, 21], 'list of the locations')

FLAGS = tf.app.flags.FLAGS

'''
plot figures of the 1-dim array
'''
def plotfigure(array, mode = 'train'):
    if mode == 'train':
        ylabel = 'RMSE'
    elif mode == 'test':
        ylabel = 'NRMSE: accuracy of testing'
    t = np.arange(np.size(array))
    plt.figure()
    plt.plot(t, array, '-')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid(color='r', linestyle='--')
    plt.show()

def plotObs(pred, obs):
    t = np.arange(np.size(pred))
    plt.figure()
    plt.plot(t, obs)
    plt.show()
'''
nrmse evaluation function
'''
def nrmse(p, o):
    N = np.size(p)
    return 100 / np.std(o) * np.sqrt(np.mean(np.multiply(p-o, p-o)))

'''
Normalization method: linear normalization and normal normalization
'''
def normalization(array):
    if FLAGS.normalization_mode == 'linear':
        for i in range(np.size(array, axis=1)):
            array[:, i] = (array[:, i] - min(array[:, i])) / (max(array[:, i]) - min(array[:, i]))
    elif FLAGS.normalization_mode == 'normal':
        for i in range(np.size(array, axis=1)):
            array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, 1])
    return array


'''
class DataHandler(object):
    def __init__(self, locations):
        self.__locations = locations

    def get_dataset(self):
        # input
        decay_rates = ('Scheme2-12.3a/', 'Scheme2-5a/', 'Scheme2-250d/', 'Scheme2-70d/')
        if isinstance(self.__locations, list):
            dye = np.empty((0, np.size(self.__locations)+4), dtype=np.float32)
            for decay_rate in decay_rates:
                for location in self.__locations:
                    dye_dir = FLAGS.folder_dir + decay_rate + 'DYETS%03d.OUT' % location
                    dye_location = np.mean(np.loadtxt(dye_dir, skiprows=4)[:, 1:], axis=1)[:, np.newaxis]
                    if self.__locations.index(location) == 0:
                        dye_temp = np.empty((np.size(dye_location, axis=0), 0), dtype=np.float32)
                    dye_temp = np.hstack((dye_temp, dye_location))
                labels = np.zeros((np.size(dye_temp, axis=0), 4))
                labels[:, decay_rates.index(decay_rate)] = 1
                dye_temp = np.hstack((dye_temp, labels))
                dye = np.vstack((dye, dye_temp))[5000:, :]
            dye[:, 0:-4] = normalization(dye[:, 0:-4])
            data_set = dye

        if FLAGS.with_overlap:
            overlapped_size = int(np.floor(FLAGS.timestep_size * FLAGS.overlapped_percent))
            sample_num = int((np.size(data_set, axis=0) - FLAGS.timestep_size) // (FLAGS.timestep_size - overlapped_size))
            samples = data_set[0:FLAGS.timestep_size][np.newaxis, :, :]
            for i in np.arange(1, sample_num):
                index = i*(FLAGS.timestep_size - overlapped_size)
                samples = np.concatenate((samples, data_set[index:(index+FLAGS.timestep_size), :][np.newaxis, :, :]),axis=0)
        else:
            samples = np.reshape(data_set, (-1, FLAGS.timestep_size, np.size(data_set,axis=1)))
        np.random.shuffle(samples)
        train_size = int(np.floor(np.size(samples, axis=0)) * (1 - FLAGS.test_percent))
        train_data = samples[0:train_size, :, :]
        test_data = samples[train_size:, :, :]
        return {'train_data': train_data,
                'test_data': test_data
                }
'''

def build_graph():
    x = tf.placeholder(tf.float32, [None, FLAGS.timestep_size, FLAGS.input_size])
    y = tf.placeholder(tf.float32, [None, FLAGS.class_num])

    # gate input to hidden layer

    gate_W = tf.Variable(tf.truncated_normal([FLAGS.input_size, FLAGS.hidden_dim], stddev=0.1), dtype=tf.float32)
    gate_bias = tf.Variable(tf.random_normal([FLAGS.hidden_dim]), dtype=tf.float32)
    x_slice1 = tf.slice(x, [0, 1, 0], [FLAGS.batch_size, FLAGS.timestep_size - 1, FLAGS.input_size])
    x_slice2 = tf.slice(x, [0, 0, 0], [FLAGS.batch_size, FLAGS.timestep_size - 1, FLAGS.input_size])
    x_slice = tf.subtract(x_slice1, x_slice2)
    x_append = tf.reduce_mean(x_slice, axis=1, keepdims=True)
    x_gate = tf.concat([x_append, x_slice], axis=1)
    alpha = tf.Variable(initial_value=10, dtype=tf.float32)
    x_gate_fold = tf.reshape(x_gate, [-1, FLAGS.input_size]) * alpha
    gate_op0 = tf.nn.sigmoid(tf.matmul(x_gate_fold, gate_W) + gate_bias)

    # nn
    W1 = tf.Variable(tf.truncated_normal([FLAGS.input_size, FLAGS.hidden_dim], stddev=0.1), dtype=tf.float32)
    bias1 = tf.Variable(tf.random_normal([FLAGS.hidden_dim]), dtype=tf.float32)
    W2 = tf.Variable(tf.truncated_normal([FLAGS.hidden_dim, FLAGS.class_num], stddev=0.1), dtype=tf.float32)
    bias2 = tf.Variable(tf.random_normal([FLAGS.class_num]), dtype=tf.float32)
    x_fold = tf.reshape(x, [-1, FLAGS.input_size])
    hidden_layer = tf.nn.tanh(tf.multiply(tf.matmul(x_fold, W1) + bias1, gate_op0))

    # gate hidden layer to output layer
    gate_W_h = tf.Variable(tf.truncated_normal([FLAGS.hidden_dim, FLAGS.class_num], stddev=0.1), dtype=tf.float32)
    gate_bias_h = tf.Variable(tf.random_normal([FLAGS.class_num]), dtype=tf.float32)
    x_slice1_h = tf.slice(tf.reshape(hidden_layer, [-1, FLAGS.timestep_size, FLAGS.hidden_dim]), [0, 1, 0], [FLAGS.batch_size, FLAGS.timestep_size - 1, FLAGS.hidden_dim])
    x_slice2_h = tf.slice(tf.reshape(hidden_layer, [-1, FLAGS.timestep_size, FLAGS.hidden_dim]), [0, 0, 0], [FLAGS.batch_size, FLAGS.timestep_size - 1, FLAGS.hidden_dim])
    x_slice_h = tf.subtract(x_slice1_h, x_slice2_h)
    x_append_h = tf.reduce_mean(x_slice_h, axis=1, keepdims=True)
    x_gate_h = tf.concat([x_append_h, x_slice_h], axis=1)
    alpha_h = tf.Variable(initial_value=1, dtype=tf.float32)
    x_gate_fold_h = tf.reshape(x_gate_h, [-1, FLAGS.hidden_dim]) * alpha_h
    gate_op_h = tf.nn.sigmoid(tf.matmul(x_gate_fold_h, gate_W_h) + gate_bias_h)

    output_layer = tf.matmul(hidden_layer, W2) + bias2

    output_layer1 = tf.multiply(gate_op_h, output_layer)

    output_layer_reshape = tf.reshape(output_layer1, [-1, FLAGS.timestep_size, FLAGS.class_num])

    logits = tf.reduce_mean(output_layer_reshape, axis=1, keepdims=False)

    y_predict = tf.nn.softmax(logits)
    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss_op)
    # evaluate the model
    correct_pred = tf.equal(tf.argmax(y_predict, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return {'loss_op': loss_op,
            'train_op': train_op,
            'test_accuracy': accuracy,
            'y_predict': y_predict,
            'x': x,
            'y': y}


'''
The train procedure
'''


def train():
    print('Begin training')
    train_loss = np.zeros(FLAGS.epochs)
    test_accuracy = np.zeros(FLAGS.epochs)
    # datahandler = DataHandler([1, 3, 5, 15, 17, 20, 21])
    # samples = datahandler.get_dataset()
    # train_data = samples['train_data']
    # test_data = samples['test_data']
    train_data = np.load('./flowrate30/train_data_'+str(FLAGS.timestep_size)+'.npy')
    test_data = np.load('./flowrate30/test_data_'+str(FLAGS.timestep_size)+'.npy')
    print(np.size(train_data, axis=0)*np.size(train_data, axis=1))
    print(np.size(test_data, axis=0)*np.size(test_data, axis=1))
    graph = build_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in np.arange(FLAGS.epochs):
            print(epoch)
            loss_each_epoch = 0
            accuracy_each_epoch_for_test = 0
            y_predict = np.array([])
            observations = np.array([])
            batch_num = int(np.size(train_data, axis=0) // FLAGS.batch_size)
            for batch_index in np.arange(batch_num - 1):
                x = train_data[batch_index * FLAGS.batch_size:(batch_index + 1) * FLAGS.batch_size, :, :-4]
                y = np.reshape(train_data[batch_index * FLAGS.batch_size:(batch_index + 1) * FLAGS.batch_size, -1, -4:],
                               newshape=(FLAGS.batch_size, FLAGS.class_num))
                loss_each_epoch += sess.run(graph['loss_op'], feed_dict={graph['x']: x, graph['y']: y}) / (
                FLAGS.batch_size * batch_num)
                sess.run(graph['train_op'], feed_dict={graph['x']: x, graph['y']: y})

                # print(sess.run(graph['alpha'], feed_dict={graph['x']: x, graph['y']: y}))

            if FLAGS.test:
                batch_num = int(np.size(test_data, axis=0) // FLAGS.batch_size)
                for batch_index in np.arange(batch_num ):
                    x_test = test_data[batch_index * FLAGS.batch_size:(batch_index + 1) * FLAGS.batch_size, :, :-4]
                    y_test = np.reshape(
                        test_data[batch_index * FLAGS.batch_size:(batch_index + 1) * FLAGS.batch_size, -1, -4:],
                        newshape=(FLAGS.batch_size, FLAGS.class_num))
                    accuracy_each_epoch_for_test += sess.run(graph['test_accuracy'], feed_dict={graph['x']: x_test,
                                                                                                graph['y']: y_test}) / batch_num

            train_loss[epoch] = np.mean(loss_each_epoch)
            test_accuracy[epoch] = accuracy_each_epoch_for_test

    return {'train_loss': train_loss,
            'test_accuracy': test_accuracy,
            }


'''
The main function
'''


def main(argv=None):
    start = time.time()
    result = train()
    train_loss = result['train_loss']
    test_accuracy = result['test_accuracy']
    elapsed = time.time() - start
    plotfigure(train_loss)
    plotfigure(test_accuracy, 'test')
    print(test_accuracy)
    np.savetxt('./train_results/DGNN/train_loss' + str(FLAGS.batch_size) + '_' + str(FLAGS.hidden_dim) + '_' + str(
        FLAGS.lr) + '.txt', train_loss)
    np.savetxt('./train_results/DGNN/test_accuracy' + str(FLAGS.batch_size) + '_' + str(FLAGS.hidden_dim) + '_' + str(
        FLAGS.lr) + '.txt', test_accuracy)
    print(elapsed)

if __name__ == '__main__':
    tf.app.run()

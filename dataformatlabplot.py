#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a lstm netwrok to predict nuclide concentration'


__author__ = 'Zhangyijing'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

tf.app.flags.DEFINE_float('lr', 5e-3, 'the learning rate')
tf.app.flags.DEFINE_float('overlapped_percent', 0.7, 'the percentage of overlapped data')
tf.app.flags.DEFINE_float('test_percent', 0.3, 'the percentage of validation data set')
tf.app.flags.DEFINE_boolean('with_overlap', True, 'whether overlapping or not')
tf.app.flags.DEFINE_boolean('test', True, 'whether test the LSTM or not')
tf.app.flags.DEFINE_string('normalization_mode', 'normal', 'normalization method: linear, normal, etc')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the size of each batch')
tf.app.flags.DEFINE_integer('input_size', 3, 'the size of the input')
tf.app.flags.DEFINE_integer('timestep_size', 24, 'the length of the truncated LSTM chain')
tf.app.flags.DEFINE_integer('hidden_dim', 64, 'the dimensions of the state, i.e., the hidden layer')
tf.app.flags.DEFINE_integer('layer_num', 1, 'the number of the stacker LSTM cells')
tf.app.flags.DEFINE_integer('class_num', 4, 'for prediction problem, the class number equals 1')
tf.app.flags.DEFINE_integer('epochs', 200, 'iteration epochs')
tf.app.flags.DEFINE_integer('step', 48, 'sampling time')
tf.app.flags.DEFINE_integer('length', 23000, ' ')
# tf.app.flags.DEFINE_string('folder_dir', '/Volumes/FAT/项目/EFDC/数据/data/', 'the directory of the data file')
tf.app.flags.DEFINE_string('folder_dir', 'E:/EFDC/EFDC computing/', 'the directory of the data file')
tf.app.flags.DEFINE_string('decay_rate', 'Scheme2-12.3a/', 'decay rate be one of 0, 70d, 250d, 5a, 12.3a')
# tf.app.flags.DEFINE_list('locations', [1, 3, 5, 15, 17, 20, 21], 'list of the locations')
tf.app.flags.DEFINE_list('locations', [1,2,3], 'list of the locations')

FLAGS = tf.app.flags.FLAGS

# delays = {1:750, 5:360, 15:900, 20:13, 17:0}
# # starts = {1:200, 5:1525, 15:1497, 20:1520, 17:0}
delays = {1:360, 2:900, 3:13}
starts = {1:1525, 2:1497, 3:1520}

def normalization(array):
    if FLAGS.normalization_mode == 'linear':
        for i in range(np.size(array, axis=1)):
            array[:, i] = (array[:, i] - min(array[:, i])) / (max(array[:, i]) - min(array[:, i]))
    elif FLAGS.normalization_mode == 'normal':
        for i in range(np.size(array, axis=1)):
            array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, 1])
    return array

class DataHandler(object):
    def __init__(self, locations):
        self.__locations = locations

    def get_dataset(self):
        # input
        decay_rates = ('Scheme2-12.3a', 'Scheme2-5a', 'Scheme2-250d', 'Scheme2-70d')
        if isinstance(self.__locations, list):
            dye = np.empty((0, np.size(self.__locations)+4), dtype=np.float32)
            for decay_rate in decay_rates:
                for location in self.__locations:
                    dye_dir = FLAGS.folder_dir + decay_rate + '/ DYETS%03d.OUT' % location
                    dye_locations = np.mean(np.loadtxt(dye_dir, skiprows=4)[:, 1:], axis=1)
                    dye_locations = np.vstack([np.loadtxt(dye_dir, skiprows=4)[:, 0], dye_locations])
                    dye_locations = np.transpose(dye_locations)
                    sio.savemat('./flowrate30/data_'+str(location)+'_'+str(decay_rate)+'.mat',{'data':dye_locations})



def main(_):
    # locations = [5, 15, 20]
    locations = [1, 2, 3]
    datahandler = DataHandler(locations)
    datahandler.get_dataset()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
    tf.app.run()
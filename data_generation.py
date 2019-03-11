#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a lstm netwrok to predict nuclide concentration'


__author__ = 'Zhangyijing'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
        decay_rates = ('Scheme2-12.3a/', 'Scheme2-5a/', 'Scheme2-250d/', 'Scheme2-70d/')
        if isinstance(self.__locations, list):
            dye = np.empty((0, np.size(self.__locations)+4), dtype=np.float32)
            for decay_rate in decay_rates:
                for location in self.__locations:
                    dye_dir = FLAGS.folder_dir + decay_rate + 'DYETS%03d.OUT' % location
                    dye_locations = np.mean(np.loadtxt(dye_dir, skiprows=4)[:, 1:], axis=1)[:, np.newaxis]
                    dye_location = dye_locations[delays[location]+starts[location]:delays[location]+starts[location]+FLAGS.length, :]
                    # dye_location = dye_location1[:-1, :] - dye_location1[1:, :]
                    if self.__locations.index(location) == 0:
                        dye_temp = np.empty((np.size(dye_location, axis=0), 0), dtype=np.float32)
                    dye_temp = np.hstack((dye_temp, dye_location))
                labels = np.zeros((np.size(dye_temp, axis=0), 4))
                labels[:, decay_rates.index(decay_rate)] = 1
                dye_temp = np.hstack((dye_temp, labels))
                dye = np.vstack((dye, dye_temp))
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
            axis0 = np.size(data_set, axis=0) % FLAGS.timestep_size
            if axis0 == 0:
                samples = np.reshape(data_set, (-1, FLAGS.timestep_size, np.size(data_set, axis=1)))
            else:
                samples = np.reshape(data_set[:-axis0, :], (-1, FLAGS.timestep_size, np.size(data_set, axis=1)))
        np.random.shuffle(samples)
        train_size = int(np.floor(np.size(samples, axis=0)) * (1 - FLAGS.test_percent))
        train_data = samples[0:train_size, :, :]
        test_data = samples[train_size:, :, :]
        np.save('train_data_'+str(FLAGS.timestep_size)+'.npy', train_data)
        np.save('test_data_'+str(FLAGS.timestep_size)+'.npy', test_data)
        return {'train_data': train_data,
                'test_data': test_data
                }

def main(_):
    # locations = [5, 15, 20]
    locations = [1, 2, 3]
    datahandler = DataHandler(locations)
    samples = datahandler.get_dataset()
    train_data = samples['train_data']
    test_data = samples['test_data']
    np.save('./flowrate30/train_data_'+str(FLAGS.timestep_size)+'.npy', train_data)
    np.save('./flowrate30/test_data_'+str(FLAGS.timestep_size)+'.npy', test_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
    tf.app.run()
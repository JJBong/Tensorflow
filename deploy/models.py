import pymysql
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import sys, os
from time import localtime, strftime
import binascii
from config import *

class RNN:
    def __init__(self, name="rnn"):
        self.net_name = name
        self.epoch_list                  = []
        self.train_error_value_list      = []
        self.validation_error_value_list = []
        self.test_error_value_list       = []
        self.test_accuracy_list          = []
        self.classnames                  = []
        self.classified_train_error_value_list = dict()
        self.classified_test_accuracy_list = dict()
        self.data = None
        self.best_accuracy = 0
        self.best_epoch = 0
        self.num_gpu = 0
        self.n_classes = None
        self.model_id = None
        self.connection = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset=charset)

    def setData(self, n_input, n_classes, n_train_data, n_test_data, flow_num_packets, data, class_names, model_id):
        self.n_input        = n_input
        self.n_classes      = n_classes
        self.n_train_data   = n_train_data
        self.n_test_data    = n_test_data


        self.n_steps        = flow_num_packets
        self.n_state_units  = n_input * 2
        self.data           = data
        self.trainData      = data.train
        self.validationData = data.validation
        self.testData       = data.test
        self.model_id       = model_id

        for name in class_names:
            self.classnames.append(name)
            self.classified_train_error_value_list[name] = list()
            self.classified_test_accuracy_list[name] = list()

        #TrainData and testData are splited with classified data for calculation of classified train error and test accuracy.
        self.classified_trainData = dict()
        self.classified_trainDataLabel = dict()
        self.classified_testData = dict()
        self.classified_testDataLabel = dict()
        for name in self.classnames:
            self.classified_trainData[name] = []
            self.classified_trainDataLabel[name] = []
            self.classified_testData[name] = []
            self.classified_testDataLabel[name] = []

        t_data_gen, t_label_gen = self.data.generator_train_data(), self.data.generator_train_label()
        for _ in range(self.n_train_data):
            t_data = t_data_gen.__next__()
            t_label = t_label_gen.__next__()
            n = np.argmax(t_label)
            self.classified_trainData[self.classnames[n]].append(t_data)
            self.classified_trainDataLabel[self.classnames[n]].append(t_label)

        t_data_gen, t_label_gen = self.data.generator_test_data(), self.data.generator_test_label()
        for _ in range(self.n_test_data):
            t_data = t_data_gen.__next__()
            t_label = t_label_gen.__next__()
            n = np.argmax(t_label)
            self.classified_testData[self.classnames[n]].append(t_data)
            self.classified_testDataLabel[self.classnames[n]].append(t_label)

    def setDataForBackTest(self, info):
        self.n_input = info.n_input_rnn
        self.bit_num = info.bit_num_r
        self.n_steps = info.n_steps
        self.data = info.flow
        self.class_names = info.split_rnn_names
        self.n_classes = len(info.split_rnn_names)
        self.n_state_units = 2 * self.n_input

    #Create model
    def makeModel(self, learning_rate=0.01):
        self.learning_rate   = learning_rate
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_input])
        self.xt = tf.transpose(self.x, [1, 0, 2])
        self.xr = tf.reshape(self.xt, [-1, self.n_input])
        self.xs = tf.split(self.xr, self.n_steps, 0)

        with tf.variable_scope(self.net_name, reuse=False):
            self.rnn_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_state_units, forget_bias=1.0)
            self.dropout_rnn_cell_1 = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_1, input_keep_prob=0.7,
                                                                    output_keep_prob=0.7)
            self.rnn_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_state_units, forget_bias=1.0)
            self.dropout_rnn_cell_2 = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_2, input_keep_prob=0.7,
                                                                    output_keep_prob=0.7)
            self.rnn_cell_3 = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_state_units, forget_bias=1.0)
            self.dropout_rnn_cell_3 = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_3, input_keep_prob=0.7,
                                                                    output_keep_prob=0.7)
            self.multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [self.dropout_rnn_cell_1, self.dropout_rnn_cell_2, self.dropout_rnn_cell_3])
            self.outputs, self.state = tf.contrib.rnn.static_rnn(self.multi_rnn_cell, self.xs, dtype=tf.float32)

        with tf.variable_scope(self.net_name, reuse=False):
            self.W = tf.Variable(tf.random_normal([self.n_state_units, self.n_classes]))
            self.B = tf.Variable(tf.random_normal([self.n_classes]))

            self.pred = tf.matmul(self.outputs[-1], self.W) + self.B
            self.prediction_y = tf.argmax(self.pred, 1)
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])

    def makeErrorAndOptimizer(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.error)

        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def learning(self, batch_size, training_epochs):
        self.batch_size      = batch_size
        self.training_epochs = training_epochs
        self.max_test_accuracy = 0
        self.init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(self.init)
            self.total_batch = int(math.ceil(self.n_train_data/float(self.batch_size)))
            print("Total batch: %d" % self.total_batch)

            for epoch in range(self.training_epochs):
                for i in range(self.total_batch):
                    batch_flows, batch_labels = self.data.next_train_batch(self.batch_size)
                    #batch_flows = batch_flows.reshape((self.batch_size, self.n_steps, self.n_input))
                    _ = sess.run(self.optimizer,
                                feed_dict={self.x: batch_flows, self.y: batch_labels})

                self.epoch_list.append(epoch)

                error_value = sess.run(self.error,
                                      feed_dict={self.x: self.trainData['packets'], self.y: self.trainData['labels']})
                self.train_error_value_list.append(error_value)

                v_error_value = sess.run(self.error,
                                         feed_dict={self.x: self.validationData['packets'], self.y: self.validationData['labels']})
                self.validation_error_value_list.append(v_error_value)

                t_accuracy_value, t_error_value = sess.run((self.accuracy, self.error),
                                                   feed_dict={self.x: self.testData['packets'], self.y: self.testData['labels']})
                self.test_error_value_list.append(t_error_value)
                self.test_accuracy_list.append(t_accuracy_value)

                self.insert_to_db(epoch, error_value, v_error_value, t_error_value, t_accuracy_value)

                print()
                print("epoch: %d, train_error_value: %f, validation_error_value: %f, test_accuracy: %f" % (epoch, error_value,  v_error_value, t_accuracy_value))

                accuracy_value_list = []
                error_value_list = []
                for name in self.classnames:
                    accuracy_value, error_value = sess.run((self.accuracy, self.error),
                                                           feed_dict={self.x: self.classified_testData[name],
                                                                      self.y: self.classified_testDataLabel[name]})
                    self.classified_test_accuracy_list[name].append(accuracy_value)
                    accuracy_value_list.append(accuracy_value)
                    error_value_list.append(error_value)

                for i in range(self.n_classes):
                    print(self.classnames[i] + ": %f  " % accuracy_value_list[i], end='')
                    print()

                if epoch > 0:
                    if self.test_accuracy_list[-1] >= self.max_test_accuracy:
                        self.best_accuracy = self.test_accuracy_list[-1]
                        self.max_test_accuracy = self.test_accuracy_list[-1]
                        self.best_epoch = epoch
                        self.saveModel(session = sess, path = packet_flow_files_dir + str(self.model_id) + "/save_rnn_model/" + 'RNN_model.ckpt')
            print()
            print("Best Epoch: {:4d}, Best Accuracy: {:10.8f}".format(self.best_epoch, self.best_accuracy))
            self.update_model_info()
            #self.drawErrorValues()

            self.drawFalsePrediction(sess, 16)

            print("RNN Training & Testing Finished!")
            print()


    def drawErrorValues(self):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.plot(self.epoch_list, self.train_error_value_list, 'r', label='Train')
        plt.plot(self.epoch_list, self.validation_error_value_list, 'g', label='Validation')
        plt.ylabel('Total Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(122)
        plt.plot(self.epoch_list, self.test_accuracy_list, 'b', label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.yticks(np.arange(min(self.test_accuracy_list), max(self.test_accuracy_list), 0.05))
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.show()

    def drawFalsePrediction(self, sess, numPrintImages):
        ground_truth = sess.run(tf.argmax(self.y, 1), feed_dict={self.y: self.testData['labels']})
        prediction = sess.run(tf.argmax(self.pred, 1), feed_dict={self.x: self.testData['packets']})

    def saveModel(self, session, path, src_scope_name='rnn'):
        if not os.path.isdir(packet_flow_files_dir + str(self.model_id) + "/save_rnn_model/"):
            os.makedirs(packet_flow_files_dir + str(self.model_id) + "/save_rnn_model/")
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        tf.train.Saver(src_vars).save(session, path)

        print("Model saved successfully")

    def insert_to_db(self, epoch, error_value, v_error_value, t_error_value, t_accuracy_value):
        insert_sql = "insert into train_info (model_id, type, epoch, test_error, test_accuracy, train_error, val_error) values (%s, %s, %s, %s, %s, %s, %s)"
        cursor = self.connection.cursor()
        cursor.execute(insert_sql, (self.model_id, 2, epoch, float(t_error_value), float(t_accuracy_value), float(error_value), float(v_error_value)))
        cursor.close()
        self.connection.commit()

    def update_model_info(self):
        end_training_time = strftime("%Y-%m-%d %I:%M:%S", localtime())
        update_model_sql = "update model_info set end_time_r=%s, best_epoch_r=%s, best_accuracy_r=%s where id = %s"
        cursor = self.connection.cursor()
        cursor.execute(update_model_sql, (end_training_time, self.best_epoch, float(self.best_accuracy), self.model_id))
        cursor.close()
        self.connection.commit()

    def predict(self, sess):
        #(munged, process) are payload_list and process_list
        munged = MungingFromDB(data=self.data, bit_num=self.bit_num, data_size_limit=self.n_input).packet_munging_for_rnn()
        return self.class_names[int(sess.run(self.prediction_y, feed_dict={self.x: munged}))]


class CNN:
    def __init__(self, name="cnn"):
        self.net_name = name
        self.epoch_list = []
        self.train_error_value_list = []
        self.validation_error_value_list = []
        self.test_error_value_list = []
        self.test_accuracy_list = []
        self.test_error_value = 0
        self.test_accuracy_value = 0
        self.classified_test_precision_list = dict()
        self.classified_test_recall_list = dict()
        self.classnames = []
        self.best_accuracy = 0
        self.max_test_accuracy = 0
        self.best_epoch = 0
        self.n_classes = None
        self.model_id = None
        self.connection = pymysql.connect(host=host, port=port, user=user, password=password, db=db,
                                          charset=charset)

    def setData(self, model_id, n_input, n_classes, n_train_data, n_validation_data, n_test_data, data, names):
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_train_data = n_train_data
        self.n_validation_data = n_validation_data
        self.n_test_data = n_test_data
        self.data = data
        self.trainData = data.train
        self.validationData = data.validation
        self.testData = data.test
        self.classnames = names
        self.model_id = model_id
        for i in range(n_classes):
            self.classified_test_precision_list[i] = list()
            self.classified_test_recall_list[i] = list()

    def setDataForBackTest(self, info):
        self.data = info.packet
        self.class_names = info.split_cnn_names
        self.n_input = info.n_input_cnn
        self.bit_num = info.bit_num_c
        self.n_classes = len(info.split_cnn_names)

    # Create model
    def makeModel(self):
        with tf.variable_scope(self.net_name):
            # tf Graph input
            self.x = tf.placeholder("float", [None, self.n_input])
            self.y = tf.placeholder("float", [None, self.n_classes])

            self.image_x_size = math.sqrt(self.n_input)
            self.image_y_size = math.sqrt(self.n_input)

            if (self.image_x_size != self.image_y_size):
                sys.exit(1)
            self.x_image = tf.reshape(self.x, [-1, int(self.image_x_size), int(self.image_y_size), 1])
            # print(self.x_image.get_shape())

            self.keep_prob = tf.placeholder(tf.float32)
            self.W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
            self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            self.h_conv1 = tf.nn.relu(
                tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv1)

            self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding='SAME')
            self.d_h_pool1 = tf.nn.dropout(self.h_pool1, keep_prob=self.keep_prob)
            # print(self.h_conv1)
            # print(self.h_pool1)
            # print(self.d_h_pool1)

            self.W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
            self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            self.h_conv2 = tf.nn.relu(
                tf.nn.conv2d(self.d_h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2)
            self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding='SAME')
            self.d_h_pool2 = tf.nn.dropout(self.h_pool2, keep_prob=self.keep_prob)  # shape=(?, 5, 5, 64)
            # print(self.h_conv2)
            # print(self.h_pool2)
            # print(self.d_h_pool2)
            last_size_x = int(self.d_h_pool2.get_shape()[1])
            last_size_y = int(self.d_h_pool2.get_shape()[2])

            self.h_pool2_flat = tf.reshape(self.d_h_pool2, [-1, last_size_x * last_size_y * 64])
            self.W_fc1 = tf.Variable(tf.truncated_normal([last_size_x * last_size_y * 64, 1024], stddev=0.1))
            self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
            self.d_h_fc1 = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)
            self.W_fc2 = tf.Variable(tf.truncated_normal([1024, self.n_classes], stddev=0.1))
            self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.n_classes]))
            self.pred = tf.matmul(self.d_h_fc1, self.W_fc2) + self.b_fc2
            self.prediction_y = tf.argmax(self.pred, 1)

    # Create model
    def makeErrorAndOptimizer(self, learning_rate=0.01):
        with tf.variable_scope(self.net_name):
            # Calculate accuracy with a Test model
            self.learning_rate = learning_rate
            self.error = tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)
            self.mean_error = tf.reduce_mean(self.error)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)
            self.prediction_ground_truth = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction_ground_truth, tf.float32))



    def learning(self, batch_size, training_epochs):
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(self.init)
            self.total_batch = int(math.ceil(self.n_train_data / float(self.batch_size)))
            print("Total batch: %d" % self.total_batch)
            self.validation_batch = int(math.ceil(self.n_validation_data / float(self.batch_size)))
            self.test_batch = int(math.ceil(self.n_test_data / float(self.batch_size)))

            for epoch in range(self.training_epochs):
                mean_errors = list()
                for i in range(self.total_batch):
                    batch_packets, batch_labels = self.data.next_train_batch(self.batch_size)
                    _, error_values = sess.run((self.optimizer, self.error),
                                               feed_dict={self.x: batch_packets, self.y: batch_labels,
                                                          self.keep_prob: 0.5})
                    mean_errors.append(np.mean(error_values))

                mean_v_errors = list()
                for i in range(self.validation_batch):
                    batch_packets, batch_labels = self.data.next_validation_batch(self.batch_size)
                    v_error_value = sess.run(self.mean_error,
                                             feed_dict={self.x: batch_packets, self.y: batch_labels,
                                                        self.keep_prob: 0.5})
                    mean_v_errors.append(v_error_value)

                self.epoch_list.append(epoch)
                self.train_error_value_list.append(np.mean(mean_errors))
                self.validation_error_value_list.append(np.mean(mean_v_errors))

                self.printLossAccuracyForTestData(epoch, sess)

                # Low validation error checking
                if epoch > 0:
                    if self.test_accuracy_list[-1] >= self.max_test_accuracy:
                        self.best_accuracy = self.test_accuracy_list[-1]
                        self.max_test_accuracy = self.test_accuracy_list[-1]
                        self.best_epoch = epoch
                        self.saveModel(session=sess, path=packet_flow_files_dir + str(
                            self.model_id) + "/save_cnn_model/" + 'CNN_model.ckpt')

                self.insert_to_db(epoch, np.mean(mean_errors), np.mean(mean_v_errors), self.test_error_value,
                                  self.test_accuracy_value)

            # self.drawErrorValues()
            print()
            print("Best Accuracy : %f" % (self.best_accuracy))
            print("CNN Training & Testing Finished!")
            print()
            self.update_model_info()

    def printLossAccuracyForTestData(self, epoch, sess):
        accuracy_value_list = []
        error_value_list = []

        classified_precision_list = dict()
        classified_recall_list = dict()
        for i in range(self.n_classes):
            classified_precision_list[i] = list()
            classified_recall_list[i] = list()

        for i in range(self.test_batch):
            batch_packets, batch_labels = self.data.next_test_batch(self.batch_size)
            accuracy_value, error_value, prediction_y = sess.run((self.accuracy, self.mean_error, self.prediction_y),
                                                                 feed_dict={self.x: batch_packets, self.y: batch_labels,
                                                                            self.keep_prob: 0.5})
            accuracy_value_list.append(accuracy_value)
            error_value_list.append(error_value)

            classified_precision = dict()
            classified_recall = dict()
            for i in range(self.n_classes):
                classified_precision[i] = list()
                classified_recall[i] = list()

            for label_index in range(len(batch_labels)):
                classified_recall[np.argmax(batch_labels[label_index])].append(
                    np.equal(prediction_y[label_index], np.argmax(batch_labels[label_index])))
                if np.equal(prediction_y[label_index], np.argmax(batch_labels[label_index])):
                    classified_precision[np.argmax(batch_labels[label_index])].append(True)
                else:
                    classified_precision[np.argmax(batch_labels[label_index])].append(False)
                    classified_precision[prediction_y[label_index]].append(False)

            for i in range(self.n_classes):
                classified_precision[i] = round(float(np.mean(classified_precision[i], dtype=np.float32)), 3)
                if not np.isnan(classified_precision[i]):
                    classified_precision_list[i].append(classified_precision[i])
                classified_recall[i] = round(float(np.mean(classified_recall[i], dtype=np.float32)), 3)
                if not np.isnan(classified_recall[i]):
                    classified_recall_list[i].append(classified_recall[i])

        self.test_error_value_list.append(np.mean(error_value_list))
        self.test_accuracy_list.append(np.mean(accuracy_value_list))
        self.test_error_value = np.mean(error_value_list)
        self.test_accuracy_value = np.mean(accuracy_value_list)
        # self.max_test_accuracy = max(self.test_accuracy_list)

        for i in range(self.n_classes):
            self.classified_test_precision_list[i].append(np.mean(classified_precision_list[i]))
            self.classified_test_recall_list[i].append(np.mean(classified_recall_list[i]))

        print()
        print("epoch: %d, test_error_value: %f, test_accuracy: %f" % (
        epoch, np.mean(error_value_list), np.mean(accuracy_value_list)))

        precision = 0.0
        recall = 0.0

        for i in range(self.n_classes):
            print("\t{:>14}: {:8.6f}, {:8.6f}  ".format(self.classnames[i], self.classified_test_precision_list[i][-1],
                                                        self.classified_test_recall_list[i][-1]))
            precision += self.classified_test_precision_list[i][-1]
            recall += self.classified_test_recall_list[i][-1]
        print("\t{:>14}: {:8.6f}, {:8.6f}  ".format("Mean of Values", precision / self.n_classes,
                                                    recall / self.n_classes))
        print()

    def drawErrorValues(self):
        fig = plt.figure(figsize=(20, 8))

        plt.subplot(221)
        plt.plot(self.epoch_list[1:], self.train_error_value_list[1:], 'r', label='Train')
        plt.plot(self.epoch_list[1:], self.validation_error_value_list[1:], 'g', label='Validation')
        plt.ylabel('Total Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(222)
        plt.plot(self.epoch_list[1:], self.test_accuracy_list[1:], 'b', label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.yticks(np.arange(0.2, 1.0, 0.1))
        plt.grid(True)
        plt.legend(loc='lower right')

        plt.subplot(223)

        plt.ylabel('PR & RC')
        plt.xlabel('Classes')
        X = np.arange(self.n_classes)
        best_pr = [pr[self.best_epoch] for pr in self.classified_test_precision_list.values()]
        best_rc = [rc[self.best_epoch] for rc in self.classified_test_recall_list.values()]
        plt.bar(X + 0.00, best_pr, color='b', label='PR', width=0.25)
        plt.bar(X + 0.25, best_rc, color='g', label='RC', width=0.25)
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.show()

    def saveModel(self, session, path, src_scope_name='cnn'):
        if not os.path.isdir(packet_flow_files_dir + str(self.model_id) + "/save_cnn_model/"):
            os.makedirs(packet_flow_files_dir + str(self.model_id) + "/save_cnn_model/")
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        tf.train.Saver(src_vars).save(session, path)
        print("Model saved successfully")

    def insert_to_db(self, epoch, error_value, v_error_value, t_error_value, t_accuracy_value):
        insert_sql = "insert into train_info (model_id, type, epoch, test_error, test_accuracy, train_error, val_error) values (%s, %s, %s, %s, %s, %s, %s)"
        cursor = self.connection.cursor()
        cursor.execute(insert_sql, (
        self.model_id, 1, epoch, float(t_error_value), float(t_accuracy_value), float(error_value),
        float(v_error_value)))
        cursor.close()
        self.connection.commit()

    def update_model_info(self):
        end_training_time = strftime("%Y-%m-%d %I:%M:%S", localtime())
        update_model_sql = "update model_info set end_time_c=%s, best_epoch_c=%s, best_accuracy_c=%s, status=%s where id = %s"
        cursor = self.connection.cursor()
        cursor.execute(update_model_sql,
                       (end_training_time, self.best_epoch, float(self.best_accuracy), 1, self.model_id))
        cursor.close()
        self.connection.commit()

    def predict(self, sess):
        # (munged, process) are payload_list and process_list
        munged = MungingFromDB(data=self.data, bit_num=self.bit_num,
                               data_size_limit=self.n_input).packet_munging_for_cnn()
        return self.class_names[
            int(sess.run(self.prediction_y, feed_dict={self.x: munged, self.keep_prob: 0.5}))]


class MungingFromDB:
    def __init__(self, data, bit_num, data_size_limit):
        self.data = list(data)
        self.bit_num = bit_num
        self.data_size_limit = data_size_limit
        self.munged_data = []

    def payloadtohex(self, payload):
        '''
        Takes a raw payload data and converts to string representation of hex
        '''
        return str(binascii.b2a_hex(payload), 'utf-8')

    def make_packet_identical_size(self, packet):
        current_packet_size = len(packet)
        diff = self.data_size_limit - current_packet_size
        if diff > 0:
            for _ in range(diff):
                packet += '0'
        return packet
    def make_packet_identical_size_for_nparray(self, np_array):
        current_array_size = len(np_array)
        diff = self.data_size_limit - current_array_size
        if diff > 0:
            for _ in range(diff):
                np_array = np.append(np_array, 0)
        return np_array
    def bitsOfPixel(self, array, bit_num, data_size_limit):
        if bit_num == 1:
            three_of_two = np.array([int(int(array[i], 16)/8) for i in range(0, len(array[:int(data_size_limit/4)]))], dtype=np.float16)
            two_of_two = np.array([int(int(array[i], 16)%8/4) for i in range(0, len(array[:int(data_size_limit/4)]))], dtype=np.float16)
            one_of_two = np.array([int(int(array[i], 16)%4/2) for i in range(0, len(array[:int(data_size_limit/4)]))], dtype=np.float16)
            zero_of_two = np.array([int(int(array[i], 16)%2) for i in range(0, len(array[:int(data_size_limit/4)]))], dtype=np.float16)
            synthesis = np.array([], dtype=np.float16)
            for i in range(len(three_of_two)):
                synthesis = np.append(synthesis, three_of_two[i])
                synthesis = np.append(synthesis, two_of_two[i])
                synthesis = np.append(synthesis, one_of_two[i])
                synthesis = np.append(synthesis, zero_of_two[i])
            return self.make_packet_identical_size_for_nparray(synthesis)
        elif bit_num == 2:
            upper_bits = np.array([int(array[i], 16)/4 for i in range(0, len(array[:int(data_size_limit/2)]))], dtype=np.float16)
            lower_bits = np.array([int(array[i], 16)%4 for i in range(0, len(array[:int(data_size_limit/2)]))], dtype=np.float16)
            for i in range(len(lower_bits)):
                upper_bits = np.insert(upper_bits, i*2-1, lower_bits[i])
            return self.make_packet_identical_size_for_nparray(upper_bits)
        elif bit_num == 4:
            return np.array([int(a, 16) for a in list(array)[:data_size_limit]], dtype=np.float16)
        elif bit_num == 8:
            even_n_packet = np.array([int(array[i], 16) for i in range(0, len(array[:data_size_limit*2]), 2)], dtype=np.float16)
            odd_n_packet = np.array([int(array[i], 16) for i in range(1, len(array[:data_size_limit*2]), 2)], dtype=np.float16)
            if len(even_n_packet) == len(odd_n_packet):
                for i in range(len(even_n_packet)):
                    odd_n_packet[i] += even_n_packet[i]*16.
                return self.make_packet_identical_size_for_nparray(odd_n_packet)
            elif len(even_n_packet) - 1 == len(odd_n_packet):
                for i in range(len(odd_n_packet)):
                    odd_n_packet[i] += even_n_packet[i]*16.
                odd_n_packet = np.append(odd_n_packet, even_n_packet[-1])
                return self.make_packet_identical_size_for_nparray(odd_n_packet)

    def packet_munging(self):
        for i, packet in enumerate(self.data):
            if packet == None or packet == '':
                del(self.data[i])
            else:
                self.data[i] = self.make_packet_identical_size(self.payloadtohex(packet)[2:-1])
                self.munged_data.append(np.array([int(a, 16) for a in list(self.data[i])[:self.data_size_limit]], dtype=np.float16))
        return self.munged_data

    def packet_munging_for_cnn(self):
        if self.data[8] == None:
            self.data[8] = '0'
        #print(self.data[8])
        payload = self.make_packet_identical_size(self.data[8])
        self.munged_data.append(self.bitsOfPixel(self.data[8], self.bit_num, self.data_size_limit))
        return self.munged_data

    def packet_munging_for_rnn(self):
        for packet in self.data[0]:
            if packet[8] == None:
                packet[8] = '0'
            payload = self.make_packet_identical_size(packet[8])
            self.munged_data.append(self.bitsOfPixel(payload, self.bit_num, self.data_size_limit))
        return [self.munged_data]
import tensorflow as tf
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from data import PacketData
from data_generator_version import PacketData
import pickle

num_gpu = int(sys.argv[7])


class CNN:
    def __init__(self):
        self.epoch_list                   = []
        self.train_error_value_list       = []
        self.validation_error_value_list  = []
        self.test_error_value_list        = []
        self.class_train_error_value_list = dict()
        self.test_accuracy_list           = []
        self.classnames                   = []
        self.class_test_accuracy_list     = dict()
        self.last_epoch                   = 0


    def setData(self, n_input, n_classes, n_train_data, n_test_data, data, class_names):
        with tf.device('/gpu:'+str(num_gpu)):
            self.n_input        = n_input            # 400
            self.n_classes      = n_classes          # 8
            self.n_train_data   = n_train_data      # 55000
            self.n_test_data    = n_test_data
            self.data           = data
            self.trainData      = data.train
            self.validationData = data.validation
            self.testData       = data.test
            for name in class_names:
                self.classnames.append(name)
                self.class_train_error_value_list[name] = list()
                self.class_test_accuracy_list[name] = list()
            
            #TrainData and testData are splited with classified data.
            self.class_trainData = dict()
            self.class_trainDataLabel = dict()
            self.class_testData = dict()
            self.class_testDataLabel = dict()
            
            for name in self.classnames:
                self.class_trainData[name] = []
                self.class_trainDataLabel[name] = []
                self.class_testData[name] = []
                self.class_testDataLabel[name] = []
            
            t_data_gen, t_label_gen = self.data.generator_train_data(), self.data.generator_train_label()
            for _ in range(self.n_train_data):
                t_data = t_data_gen.next()
                t_label = t_label_gen.next()
                n = np.argmax(t_label)
                
                self.class_trainData[self.classnames[n]].append(t_data)
                self.class_trainDataLabel[self.classnames[n]].append(t_label)
                    
            t_data_gen, t_label_gen = self.data.generator_test_data(), self.data.generator_test_label()
            for _ in range(self.n_test_data):
                t_data = t_data_gen.next()
                t_label = t_label_gen.next()
                n = np.argmax(t_label)
                
                self.class_testData[self.classnames[n]].append(t_data)
                self.class_testDataLabel[self.classnames[n]].append(t_label)


    #Create model
    def makeModel(self, learning_rate):
        with tf.device('/gpu:'+str(num_gpu)):
            self.learning_rate   = learning_rate
            
            #tf Graph input
            self.x = tf.placeholder("float", [None, self.n_input])
            self.y = tf.placeholder("float", [None, self.n_classes])
            
            self.image_x_size = math.sqrt(self.n_input)
            self.image_y_size = math.sqrt(self.n_input)
            
            if (self.image_x_size != self.image_y_size):
                sys.exit(1)
            self.x_image = tf.reshape(self.x, [-1, int(self.image_x_size), int(self.image_y_size), 1])
            print self.x_image.get_shape()
            
            self.keep_prob = tf.placeholder(tf.float32)
            self.W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
            self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            self.h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv1)

            self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.d_h_pool1 = tf.nn.dropout(self.h_pool1, keep_prob=self.keep_prob)
            print self.h_conv1
            print self.h_pool1
            print self.d_h_pool1


            self.W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
            self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.d_h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2)
            self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.d_h_pool2 = tf.nn.dropout(self.h_pool2, keep_prob=self.keep_prob) #shape=(?, 5, 5, 64)
            print self.h_conv2
            print self.h_pool2
            print self.d_h_pool2
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
            self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)
            
            # Calculate accuracy with a Test model
            self.prediction_ground_truth = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction_ground_truth, tf.float32))

            
    def learning(self, batch_size, training_epochs, training_batch_size):
        with tf.device('/gpu:'+str(num_gpu)):
            self.batch_size      = batch_size
            self.training_epochs = training_epochs
            self.training_batch_size = training_batch_size
            self.init = tf.global_variables_initializer()
            
        with tf.Session() as sess:
            sess.run(self.init)
            self.total_batch = int(math.ceil(self.n_train_data/float(self.batch_size)))
            self.training_batch = int(math.ceil(self.training_epochs/float(self.training_batch_size)))
            print "Total batch: %d" % self.total_batch
            
            for epoch in range(self.training_epochs):
                training_flag = 0
                error_value_list = dict()
                for name in self.classnames:
                    error_value_list[name] = []
                for e in range(self.training_batch):
                    for name in self.classnames:
                        error_value_list[name].append(sess.run(self.error, feed_dict={self.x: self.class_trainData[name][training_flag:training_flag+self.training_batch_size], self.y: self.class_trainDataLabel[name][training_flag:training_flag+self.training_batch_size], self.keep_prob: 0.5}))

                    training_flag += self.training_batch_size
                
                #Mean of error_values
                for name in self.classnames:
                    error_value_list[name] = np.mean(error_value_list[name])                      
                error_value = np.mean([error_value_list[name] for name in self.classnames])
                
                self.epoch_list.append(epoch)
                self.train_error_value_list.append(error_value)
                
                for name in self.classnames:
                    self.class_train_error_value_list[name].append(error_value_list[name])
                    
                v_error_value = sess.run(self.error,
                                         feed_dict={self.x: self.validationData['packets'], self.y: self.validationData['labels'], self.keep_prob: 0.5})
                self.validation_error_value_list.append(v_error_value)
                self.printLossAccuracyForTestData(epoch, sess)
                
                for i in range(self.total_batch):
                    batch_packets, batch_labels = self.data.next_train_batch(self.batch_size)
                    _ = sess.run( self.optimizer, feed_dict={self.x: batch_packets, self.y: batch_labels, self.keep_prob: 0.5})
                
                #Low validation error checking
                if epoch > 20:
                    if np.mean(self.validation_error_value_list[epoch-10:epoch]) > np.mean(self.validation_error_value_list[epoch-20:epoch-10]):
                        self.last_epoch = self.validation_error_value_list.index(np.min(self.validation_error_value_list[epoch-20:epoch]))
                        self.epoch_list[self.last_epoch+1:] = []
                        self.train_error_value_list[self.last_epoch+1:] = []
                        self.validation_error_value_list[self.last_epoch+1:] = []
                        self.test_error_value_list[self.last_epoch+1:] = []
                        self.test_accuracy_list[self.last_epoch+1:] = []
                        for name in self.classnames:
                            self.class_train_error_value_list[name][self.last_epoch+1:] = []
                            self.class_test_accuracy_list[name][self.last_epoch+1:] = []
                        print("There is enough training!") 
                        break
                    

            self.drawErrorValues()
            self.pickleDump(self.epoch_list, self.train_error_value_list, self.class_train_error_value_list, self.validation_error_value_list, self.test_error_value_list, self.test_accuracy_list, self.class_test_accuracy_list)
            self.drawFalsePrediction(sess, 16)
            print("Training % Test finished!")
            



    def printLossAccuracyForTestData(self, epoch, sess):
        with tf.device('/gpu:0'):
            accuracy_value, error_value = sess.run((self.accuracy, self.error),
                            feed_dict={self.x: self.testData['packets'], self.y: self.testData['labels'], self.keep_prob: 0.5})

            self.test_error_value_list.append(error_value)
            self.test_accuracy_list.append(accuracy_value)
            print "epoch: %d, test_error_value: %f, test_accuracy: %f" % (epoch, error_value, accuracy_value)
            
            n_test_flag = 0
            for classified_test_data, classified_test_label in [(self.class_testData[name], self.class_testDataLabel[name]) for name in self.classnames]:
                accuracy_value = sess.run(self.accuracy, feed_dict={self.x: classified_test_data, self.y: classified_test_label, self.keep_prob: 0.5})
                self.class_test_accuracy_list[self.classnames[n_test_flag]].append(accuracy_value)
                print self.classnames[n_test_flag] + ": %f  " % (accuracy_value),
                n_test_flag += 1
            print


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
        plt.plot(self.epoch_list[1:], self.class_train_error_value_list[self.classnames[0]][1:], 'r', label='rdp')
        plt.plot(self.epoch_list[1:], self.class_train_error_value_list[self.classnames[1]][1:], 'g', label='skype')
        plt.plot(self.epoch_list[1:], self.class_train_error_value_list[self.classnames[2]][1:], 'b', label='ssh')
        plt.plot(self.epoch_list[1:], self.class_train_error_value_list[self.classnames[3]][1:], 'black', label='bittorrent')
        plt.plot(self.epoch_list[1:], self.class_train_error_value_list[self.classnames[4]][1:], 'y', label='web')
        plt.ylabel('Classified Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        plt.subplot(224)
        plt.plot(self.epoch_list[1:], self.class_test_accuracy_list[self.classnames[0]][1:], 'r', label='rdp')
        plt.plot(self.epoch_list[1:], self.class_test_accuracy_list[self.classnames[1]][1:], 'g', label='skype')
        plt.plot(self.epoch_list[1:], self.class_test_accuracy_list[self.classnames[2]][1:], 'b', label='ssh')
        plt.plot(self.epoch_list[1:], self.class_test_accuracy_list[self.classnames[3]][1:], 'black', label='bittorrent')
        plt.plot(self.epoch_list[1:], self.class_test_accuracy_list[self.classnames[4]][1:], 'y', label='web')
        plt.ylabel('Classified Accuracy')
        plt.xlabel('Epochs')
        plt.yticks(np.arange(0.2, 1.0, 0.1))
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.show()
        
        
        

    def drawFalsePrediction(self, sess, numPrintImages):
        ground_truth = sess.run(tf.argmax(self.y, 1), feed_dict={self.y: self.testData['labels']})
        prediction = sess.run(tf.argmax(self.pred, 1), feed_dict={self.x: self.testData['packets'], self.keep_prob: 0.5})

    
    ########## pickle DumpList ##############
    #self.epoch_list                        #
    #self.train_error_value_list            #
    #self.class_train_error_value_list(dict)#
    #self.validation_error_value_list       #
    #self.test_error_value_list             #
    #self.test_accuracy_list                #
    #self.class_test_accuracy_list(dict)    #
    #########################################
    def pickleDump(self, *input_list):
        with open("/home/link/hdd4/network_deep_learning_data/pickle_for_graph/"+str(2**int(sys.argv[3]))+'bit_'+\
                  sys.argv[2]+'_'+sys.argv[4]+'_'+sys.argv[5]+'_'+sys.argv[6]+'.pickle', 'wb') as handle:
            for name in input_list:
                pickle.dump(name, handle)
            print "Pickle_Dumps are complicated"
                
    def pickleLoad(self, *input_list):
        merge_list = list()
        with open("/home/link/hdd4/network_deep_learning_data/pickle_for_graph/"+str(2**int(sys.argv[3]))+'bit_'+\
                  sys.argv[2]+'_'+sys.argv[4]+'_'+sys.argv[5]+'_'+sys.argv[6]+'.pickle', 'rb') as handle:
            for name in input_list:
                name = pickle.load(handle)
                merge_list.append(name)
            return merge_list


if __name__ == "__main__":
    tar_dir = "/home/link/hdd4/network_deep_learning_data/"
    start_line = int(sys.argv[1])
    data_size_limit = int(sys.argv[2])
    num_bit = int(sys.argv[3])
    num_train_data = int(sys.argv[4])
    num_validation_data = int(sys.argv[5])
    num_test_data = int(sys.argv[6])
    
    data = PacketData(tar_dir, start_line, data_size_limit, num_bit, num_train_data, num_validation_data, num_test_data)
    data.read_data_sets()
    
    
    cnn = CNN()
    cnn.setData(n_input = data_size_limit,
                n_classes = 5,
                n_train_data = num_train_data,
                n_test_data = num_test_data,
                data = data,
                class_names = ('rdp', 'skype', 'ssh', 'bittorrent', 'web'))
    
    cnn.makeModel(learning_rate = 0.0001)
    cnn.learning(batch_size = 100, training_epochs = 2000, training_batch_size =5000)
    
    
    


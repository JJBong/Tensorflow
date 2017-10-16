# coding: utf-8

import sys

from os.path import dirname
sys.path.append(dirname(__file__))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pymysql
import math
import sqlite3
from scapy.all import *
import sys, os
from config import *
from models import *

packets = sniff()
model_id = sys.argv[1]
packet_id = sys.argv[2]
packet_ids = (sys.argv[2],)
cnn_model_path = packet_flow_files_dir + str(model_id) + "/save_cnn_model/" + 'CNN_model.ckpt'
rnn_model_path = packet_flow_files_dir + str(model_id) + "/save_rnn_model/" + 'RNN_model.ckpt'

so_time_threshold = 3600
flow_insert_sql = "insert into flow_info (src_ip, dst_ip, src_port, dst_port, protocol, packet_count, last_timestamp, prediction) values (?, ?, ?, ?, ?, ?, ?, ?)"
select_flows_sql = "select * from flow_info"
update_flow_count_sql = "update flow_info set packet_count = {0} where id = {1}"
update_flow_prediction_sql = "update flow_info set prediction = '{0}' where id = {1}"
delete_flow_sql = "delete from flow_info where id = {0}"
flow_ids_insert_sql = "insert into flow_ids_info (flow_id, packet_id) values (?, ?)"
select_flow_ids_sql = "select packet_id from flow_ids_info where flow_id = {0}"
connection = sqlite3.connect("test.db")
cursor = connection.cursor()
connection.commit()
try:
    cursor.execute("create table flow_info (id integer primary key autoincrement, \
                                            src_ip text, \
                                            dst_ip text, \
                                            src_port integer, \
                                            dst_port integer, \
                                            protocol text, \
                                            packet_count integer, \
                                            last_timestamp integer not null, \
                                            prediction text)")
    connection.commit()
except sqlite3.OperationalError:
    pass
try:
    cursor.execute("create table flow_ids_info (id integer primary key autoincrement, \
                                            flow_id integer, \
                                            packet_id integer)")
    connection.commit()
except sqlite3.OperationalError:
    pass
#MYSQL_DB 정보
mysql_select_packet_sql = "SELECT `id`, `timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, lower(process), HEX(SUBSTR(`payload`, 1, 8)) as payload, `payload_size` FROM packet_info WHERE id = %s"
mysql_select_model_sql = "select lower(process_c), lower(process_r), payload_size_c, payload_size_r, packets, bit_num_c, bit_num_r from model_info where id = {0}"
mysql_host = "218.150.181.120"
mysql_port = 33060
mysql_user = "etri"
mysql_password = "linketri"
mysql_db = "network"
mysql_charset = "utf8"

mysql_connection = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, password=mysql_password, db=mysql_db, charset=mysql_charset)
mysql_cursor = mysql_connection.cursor()
mysql_cursor.execute(mysql_select_packet_sql, str(packet_id))
packet = mysql_cursor.fetchone()
mysql_connection.commit()


class ImportingData():
    def __init__(self, model_id, packet_id=0, packet_ids=0):
        self.connection = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, password=mysql_password, db=mysql_db, charset=mysql_charset)
        self.cursor = self.connection.cursor()

        sql = mysql_select_model_sql.format(int(model_id))
        self.cursor.execute(sql)
        self.model = self.cursor.fetchone()
        self.model = list(self.model)
        split_names = str(self.model[0]).replace("(", "").replace("'", "").replace(")", "").replace(",", "").split('|')
        self.split_cnn_names = []
        for i in range(len(split_names)-1):
            self.split_cnn_names.append(split_names[i])
        self.split_cnn_names.append('etc')
        split_names = str(self.model[1]).replace("(", "").replace("'", "").replace(")", "").replace(",", "").split('|')
        self.split_rnn_names = []
        for i in range(len(split_names)-1):
            self.split_rnn_names.append(split_names[i])
        self.split_rnn_names.append('etc')
        self.n_input_cnn = self.model[2]
        self.n_input_rnn = self.model[3]
        self.n_steps = self.model[4]
        self.bit_num_c = self.model[5]
        self.bit_num_r = self.model[6]
        self.connection.commit()
        self.cursor.execute(mysql_select_packet_sql, (str(packet_id)))
        self.packet = self.cursor.fetchone()
        print(self.packet)
        self.connection.commit()
        self.cursor.execute(mysql_select_packet_sql, (str(packet_id)))
        self.flow = self.cursor.fetchall()
        print(self.flow)
        self.connection.commit()
        self.cursor.close()


def restoreModel_CNN(session):
    saver = tf.train.Saver([v for v in tf.all_variables() if "cnn" in v.name])
    saver.restore(sess=session, save_path=cnn_model_path)
    
def restoreModel_RNN(session):
    saver = tf.train.Saver([v for v in tf.all_variables() if "rnn" in v.name])
    saver.restore(sess=session, save_path=rnn_model_path)

starting_flag = False
is_in_flow_info = cursor.execute("select count(*) from flow_info")
is_in_flow_info = cursor.fetchone()
connection.commit()
if is_in_flow_info[0] == 0:
    starting_flag = True
else:
    starting_flag = False
info = ImportingData(model_id=model_id, packet_id=packet_id, packet_ids=packet_ids)

if __name__ == "__main__":
    for i in range(3):
        packet = next(packets)
        print((len(packet[0]), packet[1].time, packet[1]['IP'].src, packet[1]['IP'].dst, packet[1].sport,
            packet[1].dport, packet[1].proto, "", packet[1]['UDP'].payload.load, len(packet[1]['UDP'].payload.load)))
        #process = packet[7].lower() # Ground Truth
        process = "hi"
        for key, value in process_definition_policy.items():
            if process in value:
                process = key
        with tf.Session() as sess:
            cnn = CNN()
            cnn.setDataForBackTest(info)
            cnn.makeModel()

            rnn = RNN()
            rnn.setDataForBackTest(info)
            rnn.makeModel()

            restoreModel_CNN(session=sess)
            restoreModel_RNN(session=sess)


            #flow 정보 삽입
            if starting_flag == True:
                cursor.execute(flow_insert_sql, (packet[2], packet[3], packet[4], packet[5], packet[6], 1, packet[1], ''))
                connection.commit()
                prediction = cnn.predict(sess)
                if prediction == process:
                    print('Final_Result|' + prediction + '|' + 'Success')
                else:
                    print('Final_Result|' + prediction + '|' + 'Fail')
                sys.exit()
            else:
                cursor.execute(select_flows_sql)
                flows = cursor.fetchall()
                connection.commit()
                is_RNN = False
                for flow in flows:
                    if str(flow[1]) + str(flow[2]) + str(flow[3]) + str(flow[4]) + str(flow[5]) == str(packet[2]) + str(packet[3]) + str(packet[4]) + str(packet[5]) + str(packet[6]):
                        is_RNN = True
                        if packet[1] - flow[7] <= so_time_threshold:
                            currunt_n_flow = flow[6]
                            cursor.execute(update_flow_count_sql.format(flow[6]+1, flow[0]))
                            connection.commit()
                            cursor.execute(flow_ids_insert_sql, (flow[0], packet[0]))
                            connection.commit()
                            if currunt_n_flow+1 == info.n_steps: #동일 flow packet 이 20개 쌓였을 때 --> rnn
                                global main_packet_ids
                                cursor.execute(select_flow_ids_sql.format(flow[0]))
                                main_packet_ids = cursor.fetchall()
                                if len(main_packet_ids) != info.n_steps:
                                    prediction = cnn.predict(sess)
                                    if prediction == process:
                                        print('Final_Result|' + prediction + '|' + 'Success')
                                    else:
                                        print('Final_Result|' + prediction + '|' + 'Fail')
                                    sys.exit()
                                rnn_flow = []
                                for i in main_packet_ids:
                                    mysql_cursor.execute(mysql_select_packet_sql, (i[0]))
                                    packet = mysql_cursor.fetchone()
                                    rnn_flow.append(list(packet))
                                    mysql_connection.commit()
                                info.flow = [rnn_flow]
                                rnn.setDataForBackTest(info)
                                prediction = rnn.predict(sess)
                                cursor.execute(update_flow_prediction_sql.format(prediction, flow[0]))
                                connection.commit()
                                if prediction == 'etc':
                                    prediction = cnn.predict(sess)
                                    if prediction == process:
                                        print('Final_Result|' + prediction + '|' + 'Success')
                                    else:
                                        print('Final_Result|' + prediction + '|' + 'Fail')
                                    sys.exit()
                                if prediction == process:
                                    print('Final_Result|' + prediction + '|' + 'Success')
                                else:
                                    print('Final_Result|' + prediction + '|' + 'Fail')
                                sys.exit()

                            elif currunt_n_flow+1 > info.n_steps and flow[8] == '':
                                cursor.execute(select_flow_ids_sql.format(flow[0]))
                                main_packet_ids = cursor.fetchall()
                                rnn_flow = []
                                for i in main_packet_ids:
                                    mysql_cursor.execute(mysql_select_packet_sql, (i[0]))
                                    packet = mysql_cursor.fetchone()
                                    rnn_flow.append(list(packet))
                                    mysql_connection.commit()
                                info.flow = [rnn_flow]
                                rnn.setDataForBackTest(info)
                                prediction = rnn.predict(sess)
                                cursor.execute(update_flow_prediction_sql.format(prediction, flow[0]))
                                connection.commit()
                                if prediction == 'etc':
                                    prediction = cnn.predict(sess)
                                    if prediction == process:
                                        print('Final_Result|' + prediction + '|' + 'Success')
                                    else:
                                        print('Final_Result|' + prediction + '|' + 'Fail')
                                    sys.exit()
                                if prediction == process:
                                    print('Final_Result|' + prediction + '|' + 'Success')
                                else:
                                    print('Final_Result|' + prediction + '|' + 'Fail')
                                sys.exit()

                            elif currunt_n_flow+1 > info.n_steps and flow[8] != 'etc':
                                prediction = flow[8]
                                if prediction == process:
                                    print('Final_Result|' + prediction + '|' + 'Success')
                                else:
                                    print('Final_Result|' + prediction + '|' + 'Fail')
                                sys.exit()

                            elif currunt_n_flow+1 > info.n_steps and flow[8] == 'etc':
                                prediction = cnn.predict(sess)
                                if prediction == process:
                                    print('Final_Result|' + prediction + '|' + 'Success')
                                else:
                                    print('Final_Result|' + prediction + '|' + 'Fail')
                                sys.exit()

                            else:
                                prediction = cnn.predict(sess)
                                if prediction == process:
                                    print('Final_Result|' + prediction + '|' + 'Success')
                                else:
                                    print('Final_Result|' + prediction + '|' + 'Fail')
                                sys.exit()

                        elif packet[1] - flow[7] > so_time_threshold:
                            cursor.execute(delete_flow_sql.format(flow[0]))
                            cursor.execute(flow_insert_sql, (packet[2], packet[3], packet[4], packet[5], packet[6], 1, packet[1], ''))
                            cursor.execute(flow_ids_insert_sql, (flow[0], packet[0]))
                            connection.commit()
                            prediction = cnn.predict(sess)
                            if prediction == process:
                                print('Final_Result|' + prediction + '|' + 'Success')
                            else:
                                print('Final_Result|' + prediction + '|' + 'Fail')
                            sys.exit()
                if is_RNN == False:
                    cursor.execute(flow_insert_sql, (packet[2], packet[3], packet[4], packet[5], packet[6], 1, packet[1], ''))
                    cursor.execute(flow_ids_insert_sql, (flow[0], packet[0]))
                    connection.commit()
                    prediction = cnn.predict(sess)
                    if prediction == process:
                        print('Final_Result|' + prediction + '|' + 'Success')
                    else:
                        print('Final_Result|' + prediction + '|' + 'Fail')
                    sys.exit()
        with tf.Session() as sess:
            connection.close()



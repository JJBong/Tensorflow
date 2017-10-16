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
import time
import binascii
from config2 import *
from models2 import *

packets = sniff() #sniff start
server_id = 1 #server_id
model_id = sys.argv[1]
cnn_model_path = packet_flow_files_dir + str(model_id) + "/save_cnn_model/" + 'CNN_model.ckpt'
rnn_model_path = packet_flow_files_dir + str(model_id) + "/save_rnn_model/" + 'RNN_model.ckpt'

so_time_threshold = 3600
packet_insert_sql = "insert into packet_info (timestamp, src_ip, dst_ip, src_port, dst_port, protocol, packet_size, prediction) values (?, ?, ?, ?, ?, ?, ?, ?)"
flow_insert_sql = "insert into flow_info (src_ip, dst_ip, src_port, dst_port, protocol, packet_count, last_timestamp, prediction) values (?, ?, ?, ?, ?, ?, ?, ?)"
select_flows_sql = "select * from flow_info"
result_select_packet_sql = "select prediction, count(id), sum(packet_size) from packet_info where timestamp between {0} and {1} group by prediction"
update_flow_count_sql = "update flow_info set packet_count = {0} where id = {1}"
update_flow_prediction_sql = "update flow_info set prediction = '{0}' where id = {1}"
delete_flow_sql = "delete from flow_info where id = {0}"
flow_ids_insert_sql = "insert into flow_ids_info (flow_id, packet_id) values (?, ?)"
select_flow_ids_sql = "select packet_id from flow_ids_info where flow_id = {0}"
connection = sqlite3.connect("test.db")
cursor = connection.cursor()
connection.commit()
try:
    cursor.execute("create table packet_info (id integer primary key autoincrement, \
                                            timestamp integer, \
                                            src_ip text, \
                                            dst_ip text, \
                                            src_port integer, \
                                            dst_port integer, \
                                            protocol text, \
                                            packet_size integer, \
                                            prediction text)")
    connection.commit()
except sqlite3.OperationalError:
    pass
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
mysql_result_insert_sql = "insert into deploy_server_result (server_id, timestamp, prediction, number, total_payload_size) values (%s, %s, %s, %s, %s)"
mysql_host = "218.150.181.120"
mysql_port = 33060
mysql_user = "etri"
mysql_password = "linketri"
mysql_db = "network"
mysql_charset = "utf8"

mysql_connection = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, password=mysql_password, db=mysql_db, charset=mysql_charset)
mysql_cursor = mysql_connection.cursor()


class ImportingData():
    def __init__(self, model_id):
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
        self.cursor.close()


def restoreModel_CNN(session):
    saver = tf.train.Saver([v for v in tf.all_variables() if "cnn" in v.name])
    saver.restore(sess=session, save_path=cnn_model_path)
    
def restoreModel_RNN(session):
    saver = tf.train.Saver([v for v in tf.all_variables() if "rnn" in v.name])
    saver.restore(sess=session, save_path=rnn_model_path)

def payloadtohex(payload):
        '''
        Takes a raw payload data and converts to string representation of hex
        '''
        return str(binascii.b2a_hex(payload), 'utf-8')

def packet_insert_into_sqlite(packet, prediction):
    cursor.execute(packet_insert_sql, (packet[1], packet[2], packet[3], packet[4], packet[5], packet[6], packet[-1], prediction))
    connection.commit()

def result_insert_into_mysql(pre_time, after_time):
    cursor.execute(result_select_packet_sql.format(pre_time, after_time))
    result = cursor.fetchall()
    connection.commit()
    t = time.strftime('%Y-%m-%d %H:%M:%S')
    for r in result:
        mysql_cursor.execute(mysql_result_insert_sql, (str(server_id), t, r[0], str(r[1]), str(r[2])))
        mysql_connection.commit()


#mysql db 에 넣는 사이시간 간격과 앞 뒤의 시간 그리고 시간의 초기화 플래그 변수
insert_time_interval = 10
previous_time = 0
hereafter_time = previous_time + insert_time_interval
time_init_flag = True

starting_flag = False
is_in_flow_info = cursor.execute("select count(*) from flow_info")
is_in_flow_info = cursor.fetchone()
connection.commit()
if is_in_flow_info[0] == 0:
    starting_flag = True
else:
    starting_flag = False
info = ImportingData(model_id=model_id)

if __name__ == "__main__":
    with tf.Session() as sess:
        cnn = CNN()
        cnn.setDataForBackTest(info)
        cnn.makeModel()

        rnn = RNN()
        rnn.setDataForBackTest(info)
        rnn.makeModel()

        restoreModel_CNN(session=sess)
        restoreModel_RNN(session=sess)

        while True:
            #p = (packet_lst, packet)
            #p[0] = list, p[1] = packet
            p = next(packets) #generator of sniff packets
            if time_init_flag:
                previous_time = p[1].time
                hereafter_time = previous_time + insert_time_interval
                time_init_flag = False

            #패킷 도착시간이 이전 이후 시간 사이에 존재하면 pass 그렇지 않으면 insert 후 시간 초기화
            if p[1].time >= previous_time and p[1].time < hereafter_time:
                pass
            else:
                time_init_flag = True
                result_insert_into_mysql(previous_time, hereafter_time)
            try:
                if p[1].proto == 6:
                    try:
                        packet = (len(p[0]), p[1].time, p[1]['IP'].src, p[1]['IP'].dst, p[1].sport,
                            p[1].dport, p[1].proto, "", payloadtohex(p[1]['TCP'].payload.load), len(payloadtohex(p[1]['TCP'].payload.load)))
                        print(packet)
                    except IndexError:
                        continue
                elif p[1].proto == 17:
                    try:
                        packet = (len(p[0]), p[1].time, p[1]['IP'].src, p[1]['IP'].dst, p[1].sport,
                            p[1].dport, p[1].proto, "", payloadtohex(p[1]['UDP'].payload.load), len(payloadtohex(p[1]['TCP'].payload.load)))
                        print(packet)
                    except IndexError:
                        continue
                else:
                    continue
            except AttributeError:
                continue

            cnn.setDataForBackTest(info, data=packet)
            #flow 정보 삽입
            if starting_flag == True:
                cursor.execute(flow_insert_sql, (packet[2], packet[3], packet[4], packet[5], packet[6], 1, packet[1], ''))
                connection.commit()
                prediction = cnn.predict(sess)
                print(prediction)
                packet_insert_into_sqlite(packet, prediction)
                continue
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
                                    print(prediction)
                                    packet_insert_into_sqlite(packet, prediction)
                                    continue
                                rnn_flow = []
                                for i in main_packet_ids:
                                    mysql_cursor.execute(mysql_select_packet_sql, (i[0]))
                                    packet = mysql_cursor.fetchone()
                                    rnn_flow.append(list(packet))
                                    mysql_connection.commit()
                                rnn.setDataForBackTest(info, data=[rnn_flow])
                                prediction = rnn.predict(sess)
                                cursor.execute(update_flow_prediction_sql.format(prediction, flow[0]))
                                connection.commit()
                                if prediction == 'etc':
                                    prediction = cnn.predict(sess)
                                    print(prediction)
                                    packet_insert_into_sqlite(packet, prediction)
                                    continue
                                print(prediction)
                                packet_insert_into_sqlite(packet, prediction)
                                continue

                            elif currunt_n_flow+1 > info.n_steps and flow[8] == '':
                                cursor.execute(select_flow_ids_sql.format(flow[0]))
                                main_packet_ids = cursor.fetchall()
                                rnn_flow = []
                                for i in main_packet_ids:
                                    mysql_cursor.execute(mysql_select_packet_sql, (i[0]))
                                    packet = mysql_cursor.fetchone()
                                    rnn_flow.append(list(packet))
                                    mysql_connection.commit()
                                rnn.setDataForBackTest(info, data=[rnn_flow])
                                prediction = rnn.predict(sess)
                                cursor.execute(update_flow_prediction_sql.format(prediction, flow[0]))
                                connection.commit()
                                if prediction == 'etc':
                                    prediction = cnn.predict(sess)
                                    print(prediction)
                                    packet_insert_into_sqlite(packet, prediction)
                                    continue
                                print(prediction)
                                packet_insert_into_sqlite(packet, prediction)
                                continue

                            elif currunt_n_flow+1 > info.n_steps and flow[8] != 'etc':
                                prediction = flow[8]
                                print(prediction)
                                packet_insert_into_sqlite(packet, prediction)
                                continue

                            elif currunt_n_flow+1 > info.n_steps and flow[8] == 'etc':
                                prediction = cnn.predict(sess)
                                print(prediction)
                                packet_insert_into_sqlite(packet, prediction)
                                continue

                            else:
                                prediction = cnn.predict(sess)
                                print(prediction)
                                packet_insert_into_sqlite(packet, prediction)
                                continue

                        elif packet[1] - flow[7] > so_time_threshold:
                            cursor.execute(delete_flow_sql.format(flow[0]))
                            cursor.execute(flow_insert_sql, (packet[2], packet[3], packet[4], packet[5], packet[6], 1, packet[1], ''))
                            cursor.execute(flow_ids_insert_sql, (flow[0], packet[0]))
                            connection.commit()
                            prediction = cnn.predict(sess)
                            print(prediction)
                            packet_insert_into_sqlite(packet, prediction)
                            continue
                if is_RNN == False:
                    cursor.execute(flow_insert_sql, (packet[2], packet[3], packet[4], packet[5], packet[6], 1, packet[1], ''))
                    cursor.execute(flow_ids_insert_sql, (flow[0], packet[0]))
                    connection.commit()
                    prediction = cnn.predict(sess)
                    print(prediction)
                    packet_insert_into_sqlite(packet, prediction)
                    continue
        with tf.Session() as sess:
            connection.close()



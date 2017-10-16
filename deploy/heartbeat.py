import pymysql
from urllib import request

host = "218.150.181.120"
port = 33060
user = "etri"
password = "linketri"
db = "network"
charset = "utf8"

insert_sql = "insert into server_info (model_id, name, ip_addr, deploy_time, enable) values (%s, %s, %s, now(6), %s)"

connection = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset=charset)
cursor = connection.cursor()
cursor.execute(insert_sql, (1, 'test', 'http://192.168.0.8:8000/task/', 1))
connection.commit()
cursor.close()

req =  request.Request("http://192.168.0.27:8080/deployment/heartbeat?serverId=2") # this will make the method "POST"
resp = request.urlopen(req)
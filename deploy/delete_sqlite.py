import sqlite3

connection = sqlite3.connect("test.db")
cursor = connection.cursor()
cursor.execute("drop table if exists flow_info;")
cursor.execute("drop table if exists packet_info;")
cursor.execute("drop table if exists flow_ids_info;")
connection.commit()
cursor.close()
connection.close()
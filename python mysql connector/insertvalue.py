import mysql.connector
conn=mysql.connector.connect(host='localhost', user='root', password='2323', database='pythondb')
mycursor=conn.cursor()

sql='insert into student (name,branch,id) values(%s,%s,%s)'

val=[('a','cse','56'), ('b','IT','78'), ('c','me','90')]

mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount, 'record inserted')
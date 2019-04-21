import os
import MySQLdb


db = MySQLdb.connect(host="localhost", user="root", passwd="pwd", db="fix_ghanas_potholes")
cur = db.cursor()

# Said latitude and longitude value are passed by the Nmea parser

# Transmit takes a latitude and longitude value passed to it
def transmit(latitude, longitude):
    sql = "INSERT INTO potholes (latitude, longitude) VALUES (%f,%f)"

    try:
        cur.execute(sql, latitude, longitude)
        db.commit()

    except:
        db.rollback()

    cur.close()
    db.close()

    return


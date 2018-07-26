import serial

PORT = '/dev/ttyACM1'
BAUD = 38400
ser = serial.Serial(PORT, BAUD, timeout=6)
#
# def send(cmd):
#     ser.write(str(cmd).encode('utf-8'))
#     a = ser.readline()
#     # print(a)
#     return a
#
#
# a = send('S')
# print(a)


while(ser.in_waiting):
        a = ser.readline()
        print(a)
        time.sleep(50)


ser.close()
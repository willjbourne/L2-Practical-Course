import serial
import numpy as np
import time
# Collecting the data from the microbit
serialPort = serial.Serial(port = "/dev/tty.usbmodem141102", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

d_list = []
last_data = time.time()
while(len(d_list) < 10): # (time.time()-last_data < 10) or
    # Wait until there is data waiting in the serial buffer
    if(serialPort.in_waiting > 0):
        last_data = time.time() # new data has arrived! update the time :)
        serialString = serialPort.readline() # Read data out of the buffer until a carraige return / new line is found
        d_list.append(serialString) # add serial data to the array

print(d_list)

with open("datafile.txt", 'wb') as f:
    for item in d_list:
        f.write(item[:-2])
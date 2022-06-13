import serial
import numpy as np
import time
import struct
# Collecting the data from the microbit
serialPort = serial.Serial(port = "/dev/tty.usbmodem142102", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

d_list = []
last_data = time.time()
while((time.time()-last_data < 10)): # (time.time()-last_data < 10) or
    # Wait until there is data waiting in the serial buffer
    if(serialPort.in_waiting > 0):
        last_data = time.time() # new data has arrived! update the time :)
        serialString = serialPort.readline() # Read data out of the buffer until a carraige return / new line is found
        d_list.append(serialString) # add serial data to the array

temp = d_list[0][:-2] # extract the temperature from the serial stream



with open("data/mb1/datafile-{0}-4.txt".format(temp), 'wb') as f:
    for item in d_list[1:]:
        f.write(item[:-2])

print(d_list)
print(type(d_list[0]))
print(len(d_list))
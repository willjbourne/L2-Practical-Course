import serial
import numpy as np
import time
import struct
# Collecting the data from the microbit
serialPort = serial.Serial(port = "/dev/tty.usbmodem142102", baudrate=115200, bytesize=8, timeout=5, stopbits=serial.STOPBITS_ONE)

d_list = []

# Wait until there is data in the serial buffer
while (serialPort.in_waiting <= 0):
    pass

while (serialPort.in_waiting > 0):
    serialString = serialPort.readline() # Read data out of the buffer until a carraige return / new line is found
    d_list.append(serialString) # add serial data to the array

temp = d_list[0][:-2] # extract the temperature from the serial stream



with open("data/mb4/datafile-{0}-T.txt".format(temp), 'wb') as f:
    for item in d_list:
        f.write(item)

print(d_list)
print(type(d_list[0]))
print(len(d_list))
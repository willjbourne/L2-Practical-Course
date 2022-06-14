import serial
import time

z1serial = serial.Serial(port='/dev/tty.usbmodem142102', baudrate=115200)
z1serial.timeout = 2  # set read timeout
# print z1serial  # debug serial.
print (z1serial.is_open)  # True for opened
if z1serial.is_open:
    while True:
        size = z1serial.inWaiting()
        if size:
            data = z1serial.read(size)
            print(data)
        else:
            print('no data')
        time.sleep(1)
else:
    print ('z1serial not open')
# z1serial.close()  # close z1serial if z1serial is open.
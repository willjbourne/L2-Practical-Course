% Will Bourne 14/06/22

device = serialport("/dev/tty.usbmodem142102",115200);
device.Timeout = 5;
out = read(device,999999,'uint8');
disp(out);

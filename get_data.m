<<<<<<< HEAD
device = serialport("COM12", 115200);
=======
device = serialport("/dev/tty.usbmodem142302", 115200);
>>>>>>> 5ef71f2fa250ffc5f815f438bd90c2bf4836c0da
device.Timeout = 5;
out = read(device, 999999, 'uint8');
temp = out(1:4);
temp = typecast(uint8(temp), 'int32');
t = temp / 4;
<<<<<<< HEAD
f = fopen("data/full/mb20/data5-" +  t + ".bin", "w");
=======

% delete("data/mb_query/*.bin");
f = fopen("data/full/mb19/data05-" +  t + ".bin", "w");
>>>>>>> 5ef71f2fa250ffc5f815f438bd90c2bf4836c0da
fwrite(f, out(5:size(out,2)));
fclose(f);

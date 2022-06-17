device = serialport("/dev/tty.usbmodem142302", 115200);
device.Timeout = 5;
out = read(device, 999999, 'uint8');
temp = out(1:4);
temp = typecast(uint8(temp), 'int32');
t = temp / 4;

delete("data/mb_query/*.bin");
f = fopen("data/mb_query/dataq-" +  t + ".bin", "w");
fwrite(f, out(5:size(out,2)));
fclose(f);

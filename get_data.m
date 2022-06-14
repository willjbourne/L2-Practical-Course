device = serialport("COM3", 115200);
device.Timeout = 10;
out = read(device, 999999, 'uint8');
temp = out(1:4);
temp = typecast(uint8(temp), 'int32');
t = temp / 4;
f = fopen("data/mb3/data10 -" +  t + ".bin", "w");
fwrite(f, out(4:size(out,2)));
fclose(f);
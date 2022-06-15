device = serialport("COM7", 115200);
device.Timeout = 5;
out = read(device, 999999, 'uint8');
temp = out(1:4);
temp = typecast(uint8(temp), 'int32');
t = temp / 4;
f = fopen("data/mb5/data11-" +  t + ".bin", "w");
fwrite(f, out(5:size(out,2)));
fclose(f);

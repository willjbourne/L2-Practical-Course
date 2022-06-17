device = serialport("COM27", 115200);
device.Timeout = 5;
out = read(device, 999999, 'uint8');
temp = out(1:4);
temp = typecast(uint8(temp), 'int32');
t = temp / 4;


% delete("data/mb_query/*.bin");
% mb 4 data 5
f = fopen("data/full/mb35/data5-" +  t + ".bin", "w");
% f = fopen("data/mb_query/dataq-" +  t + ".bin", "w");
fwrite(f, out(5:size(out,2)));
fclose(f);

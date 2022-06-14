void setup() {  
  Serial.begin(115200);
  
  // start temerature sensor
  *(int*)(0x4000C000 + 0x0) = 0x1;
  delay(5000);

  // read DATAREADY
  // Serial.println(*(int*)(0x4000C000 + 0x100));
  // delay(5000);

  // print temperature in °C to serial
  for (int i = 0; i < 4; i++)
  {
    Serial.print(*(char*)(0x4000C000 + 0x508 + i));
  }

  // start getting SRAM memory values
  int addy = 0x20000000;
  while (addy < 0x40000000) // upper limit of memory addresses 
  {
    // Serial.println(addy);
    Serial.print(*(char*)addy);
    addy++;
    // delay(10);
  }
}

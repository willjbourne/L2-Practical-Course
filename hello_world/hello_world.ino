const int COL1 = 3;     // Column #1 control
const int LED = 25;     // 'row 1' led


void setup() {  
  Serial.begin(9600);
  Serial.println("PUF is ready!");

  // start temerature sensor
  *(int*)(0x4000C000 + 0x0) = 0x1;
  delay(5000);

  // read DATAREADY
  // Serial.println(*(int*)(0x4000C000 + 0x100));
  // delay(5000);

  // print temperature in °C to serial
  Serial.println(0.25 * *(int*)(0x4000C000 + 0x508));


  // start getting SRAM memory values
  int addy = 0x20000000;
  while (addy < 0x40000000) // upper limit of memory addresses 
  {
    Serial.println(*(char*)addy);
    addy += 0x1;
    // delay(10);
  }
}

void loop(){
  delay(500);
  // 0x20000000 - 0x40000000
}

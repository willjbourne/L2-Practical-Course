const int COL1 = 3;     // Column #1 control
const int LED = 25;     // 'row 1' led

void setup() {  
  Serial.begin(9600);
  
  Serial.println("microbit is ready!");

  // because the LEDs are multiplexed, we must ground the opposite side of the LED
  pinMode(COL1, OUTPUT);
  digitalWrite(COL1, LOW); 
   
  pinMode(LED, OUTPUT);   

  // print temperature in °C to serial
  Serial.println(*(int*)(0x4000C000 + 0x508));

  // start getting SRAM memory values
  int addy = 0x20000000;
  while (addy < 0x40000000) // upper limit of memory addresses 
  {
    Serial.println(*(char*)addy);
    addy += 0x4;
    delay(10);
  }
}

void loop(){
  delay(500);
  // 0x20000000 - 0x40000000
}

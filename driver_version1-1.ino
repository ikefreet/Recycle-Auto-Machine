#include <Servo.h>

int servo = 4;
int dc = 3;

Servo esc;

void setup() {
  pinMode(servo, OUTPUT); // 핀모드 설정
  pinMode(dc, OUTPUT);
  esc.attach(dc);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) // 입력이 있을 때
  {
    char c = Serial.read();

    if (c == 'w') // w이면 전진
    {
      Serial.println("수동 전진");
      analogWrite(servo, 190);
      //delay(500);
      esc.write(95);
      delay(1000); 
    }

    else if (c == 'a') //a면 좌회전
    {
      Serial.println("수동 좌회전");
      analogWrite(servo, 120);
      //delay(500);
      esc.write(95);
      delay(1000);
      c="";  // 문자 비우기
    }

    else if (c == 'd') //d면 우회전
    {
      Serial.println("수동 우회전");
      analogWrite(servo, 250);
      //delay(500);
      esc.write(95);
      delay(1000);
      c="";  // 문자 비우기
    }

    else if (c == 'b') //b면 후진
    {
      Serial.println("수동 후진");
      analogWrite(servo, 190);
      esc.write(90);
      delay(500);
      esc.write(75);
      esc.write(85);
      delay(1000);
      c="";  // 문자 비우기
    }
    
    else if (c == 's') // 정지
    { 
      Serial.println("수동 정지");
      analogWrite(servo, 190);
      esc.write(90);
      delay(500);
    }
    
    else if (c == 'g')
    {
      Serial.println("자동 전진");
      analogWrite(servo, 190);
      esc.write(94);
      delay(1000);
    }
    
    else if (c == 'l') //l이면 좌회전 전진
    {
      Serial.println("자동 좌회전 전진");
      analogWrite(servo, 150); // 바퀴 모터의 각도값을 수동 때보다 축소
      //delay(500);
      esc.write(94); // 바퀴의 속도도 감소
      delay(1000);
      c="";  // 문자 비우기
    }

    else if (c == 'r') //r이면 우회전 전진
    {  
      Serial.println("자동 우회전 전진");
      analogWrite(servo, 200); // 바퀴 모터의 각도값을 수동때보다 축소
      //delay(500);
      esc.write(94); // 바퀴의 속도도 감소
      delay(1000);
      c="";  // 문자 비우기
    }

    else if (c == 'n') //n면 좌회전 후진
    {
      Serial.println("자동 좌회전 후진");
      analogWrite(servo, 200); // 바퀴 모터의 각도값을 수동때보다 축소
      //delay(500);
      esc.write(86); // 바퀴의 속도도 감소
      delay(1000);
      c="";  // 문자 비우기
    }

    else if (c == 'm') //m면 우회전 후진
    {
      Serial.println("자동 우회전 후진");
      analogWrite(servo, 200); // 바퀴 모터의 각도값을 수동때보다 축소
      //delay(500);
      esc.write(86); // 바퀴의 속도도 감소
      delay(1000);
      c="";  // 문자 비우기
    }
    
    else if (c == 't') // 정지
    { 
      Serial.println("자동 정지");
      analogWrite(servo, 190);
      esc.write(90);
      delay(500);
    }
  }
  
  else // 그 외에는 정지
  {
    analogWrite(servo, 190);
    esc.write(90);
    delay(500);
  }
}


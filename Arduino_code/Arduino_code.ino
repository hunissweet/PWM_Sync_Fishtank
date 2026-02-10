#include <Arduino.h>

#include <mbed.h>

using namespace mbed;



static DigitalOut trig(digitalPinToPinName(2));

static Ticker frameTicker;

static Timeout pulseOff;



static volatile uint32_t g_period_us = 100000; // default 1000 Hz

static volatile uint32_t g_pulse_us  = 100;   // default 100 us



static String buf;



void trig_low() { trig = 0; }



void trig_high() {

  trig = 1;

  pulseOff.attach_us(&trig_low, (int)g_pulse_us);

}



void apply_schedule() {

  frameTicker.detach();

  trig = 0;

  frameTicker.attach_us(&trig_high, (int)g_period_us);

}



void handleLine(String s) {

  s.trim();

  if (!s.length()) return;



  // allow pure number => frequency

  bool hasAlpha = false;

  for (size_t i = 0; i < s.length(); i++) {

    char c = s[i];

    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) { hasAlpha = true; break; }

  }



  float f = -1.0f;

  long p  = -1;



  if (!hasAlpha) {

    f = s.toFloat();

  } else {

    s.replace("=", " ");

    s.replace(",", " ");

    s.toUpperCase();



    int i = 0;

    while (i < (int)s.length()) {

      while (i < (int)s.length() && s[i] == ' ') i++;

      if (i >= (int)s.length()) break;



      char key = s[i++];  // F or P

      while (i < (int)s.length() && s[i] == ' ') i++;



      int start = i;

      while (i < (int)s.length() && s[i] != ' ') i++;

      String val = s.substring(start, i);



      if (key == 'F') f = val.toFloat();

      if (key == 'P') p = val.toInt();

    }

  }



  if (f > 0.0f) {

    // period in us

    uint32_t per = (uint32_t)(1000000.0f / f + 0.5f);

    if (per < 200) per = 200;           

    g_period_us = per;

  }

  if (p > 0 && p < 1000000) {

    g_pulse_us = (uint32_t)p;

  }

  if (g_pulse_us >= g_period_us) {

    g_pulse_us = g_period_us / 2;       

  }



  apply_schedule();



  Serial.print("OK F=");

  Serial.print(1000000.0f / (float)g_period_us, 3);

  Serial.print(" Hz, P=");

  Serial.print(g_pulse_us);

  Serial.println(" us");

}



void setup() {

  Serial.begin(115200);



  // 最多等 2 秒，避免卡死

  unsigned long t0 = millis();

  while (!Serial && (millis() - t0 < 2000)) {}



  trig = 0;

  apply_schedule();



  Serial.println("READY");

  Serial.println("Trigger on pin 2 (D2). Send: 100 | F 200 | P 50 | F 200 P 50");

}



void loop() {

  while (Serial.available()) {

    char c = (char)Serial.read();

    if (c == '\n') { handleLine(buf); buf = ""; }

    else if (c != '\r') { if (buf.length() < 80) buf += c; }

  }

}

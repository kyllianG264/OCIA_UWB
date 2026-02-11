#include <PortentaUWBShield.h>

uint8_t SRC_ADDR[] = {0x11, 0x11};   // Anchor1 (printer)
uint8_t DST_ADDR[] = {0x22, 0x22};   // Stella

volatile uint16_t d1_cm = 0xFFFF;
uint16_t d2_cm = 0xFFFF; // reçu de l'autre ancre via Serial1

void rangingHandler(UWBRangingData &data) {
  if (data.measureType() != (uint8_t)uwb::MeasurementType::TWO_WAY) return;

  RangingMeasures m = data.twoWayRangingMeasure();
  for (int i = 0; i < data.available(); i++) {
    if (m[i].status == 0 && m[i].distance != 0xFFFF) {
      d1_cm = m[i].distance;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("ANCHOR1 (controller) - imprime D1 + D2");

  // Lien entre ancres (RX1 reçoit Anchor2)
  Serial1.begin(115200);

  UWB.registerRangingCallback(rangingHandler);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBRangingController a1(0x12345678, src, dst);
  UWBSessionManager.addSession(a1);

  a1.init();
  a1.start();

  Serial.println("Anchor1 pret");
}

void loop() {
  // Lecture des messages venant de Anchor2 : format "D2:123\n"
  static char buf[32];
  static int idx = 0;

  while (Serial1.available()) {
    char c = (char)Serial1.read();
    if (c == '\n') {
      buf[idx] = 0;
      idx = 0;

      if (buf[0] == 'D' && buf[1] == '2' && buf[2] == ':') {
        int v = atoi(buf + 3);
        if (v > 0 && v < 65535) d2_cm = (uint16_t)v;
      }
    } else {
      if (idx < (int)sizeof(buf) - 1) buf[idx++] = c;
    }
  }

  // Print périodique
  static uint32_t t0 = 0;
  if (millis() - t0 > 200) {
    t0 = millis();

    Serial.print("D1(Stella-Anchor1)=");
    if (d1_cm == 0xFFFF) Serial.print("--");
    else Serial.print(d1_cm);
    Serial.print(" cm   ");

    Serial.print("D2(Stella-Anchor2)=");
    if (d2_cm == 0xFFFF) Serial.print("--");
    else Serial.print(d2_cm);
    Serial.println(" cm");
  }

  delay(10);
}

#include <PortentaUWBShield.h>

uint8_t SRC_ADDR[] = {0x33, 0x33};   // Anchor2 (silencieuse)
uint8_t DST_ADDR[] = {0x22, 0x22};   // Stella

volatile uint16_t d2_cm = 0xFFFF;

void rangingHandler(UWBRangingData &data) {
  if (data.measureType() != (uint8_t)uwb::MeasurementType::TWO_WAY) return;

  RangingMeasures m = data.twoWayRangingMeasure();
  for (int i = 0; i < data.available(); i++) {
    if (m[i].status == 0 && m[i].distance != 0xFFFF) {
      d2_cm = m[i].distance;
    }
  }
}

void setup() {
  // On ne print rien sur Serial USB
  Serial1.begin(115200); // TX1 va vers Anchor1 RX1

  UWB.registerRangingCallback(rangingHandler);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBRangingController a2(0x12345679, src, dst);
  UWBSessionManager.addSession(a2);

  a2.init();
  a2.start();
}

void loop() {
  static uint32_t t0 = 0;
  if (millis() - t0 > 200) {
    t0 = millis();
    if (d2_cm != 0xFFFF) {
      Serial1.print("D2:");
      Serial1.print(d2_cm);
      Serial1.print("\n");
    }
  }
  delay(10);
}

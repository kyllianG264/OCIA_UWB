#include <StellaUWB.h>

uint8_t SRC_ADDR[] = {0x22, 0x22};   // Stella
uint8_t DST_ADDR[] = {0x11, 0x11};   // Portenta

void rangingHandler(UWBRangingData &data) {
  if (data.measureType() != (uint8_t)uwb::MeasurementType::TWO_WAY)
    return;

  RangingMeasures m = data.twoWayRangingMeasure();

  for (int i = 0; i < data.available(); i++) {
    if (m[i].status == 0 && m[i].distance != 0xFFFF) {
      Serial.print("Distance = ");
      Serial.print(m[i].distance);
      Serial.println(" cm");
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("STELLA TAG");

  UWB.registerRangingCallback(rangingHandler);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBTracker tag(0x12345678, src, dst);
  UWBSessionManager.addSession(tag);

  tag.init();
  tag.start();

  Serial.println("Tag pret");
}

void loop() {
  delay(10);
}

#include <PortentaUWBShield.h>

// ===== UWB addresses =====
uint8_t SRC_ADDR[] = {0x11, 0x11};   // Portenta (Anchor Master)
uint8_t DST_ADDR[] = {0x22, 0x22};   // Stella

// ===== 10 Hz print =====
static const uint32_t CYCLE_MS = 100;

volatile uint32_t cbCount = 0;
volatile uint8_t  lastStatus = 255;
volatile uint16_t lastDist = 0xFFFF;
volatile uint16_t d1_cm = 0xFFFF;

void rangingHandler(UWBRangingData &data) {
  if (data.measureType() != (uint8_t)uwb::MeasurementType::TWO_WAY) return;

  RangingMeasures m = data.twoWayRangingMeasure();

  // Note: souvent available() = 1, mais on reste générique
  for (int i = 0; i < data.available(); i++) {
    cbCount++;
    lastStatus = m[i].status;
    lastDist   = m[i].distance;

    if (m[i].status == 0 && m[i].distance != 0xFFFF) {
      d1_cm = m[i].distance;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("ANCHOR MASTER (controller) DEBUG");

  UWB.registerRangingCallback(rangingHandler);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBRangingController anchor(0x12345678, src, dst);
  UWBSessionManager.addSession(anchor);

  anchor.init();
  anchor.start();

  Serial.println("Anchor pret");
}

void loop() {
  static uint32_t t0 = 0;
  uint32_t now = millis();

  if (now - t0 >= CYCLE_MS) {
    t0 += CYCLE_MS;

    uint32_t c = cbCount;     // copie volatile
    uint8_t  s = lastStatus;
    uint16_t ld = lastDist;
    uint16_t d1 = d1_cm;

    Serial.print("cb=");
    Serial.print(c);
    Serial.print("  lastStatus=");
    Serial.print(s);
    Serial.print("  lastDist=");
    Serial.print(ld);
    Serial.print("  D1=");
    if (d1 == 0xFFFF) Serial.print("--");
    else Serial.print(d1);
    Serial.println(" cm");
  }

  delay(1);
}

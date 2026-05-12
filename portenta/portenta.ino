#include <PortentaUWBShield.h>

// Continuous initiator for one Portenta.
// Flash the same sketch on each Portenta and only change:
// - SRC_ADDR
// - THIS_SESSION_ID
//
// Stella will rotate across the configured sessions. Each Portenta keeps
// ranging continuously and gets replies when Stella is on its session.

static const uint32_t THIS_SESSION_ID = 0x1001;
static const uint32_t PRINT_MS = 100;
static const uint32_t UART_BAUD = 115200;

uint8_t SRC_ADDR[] = {0x11, 0x11};
uint8_t DST_ADDR[] = {0x22, 0x22};

volatile uint32_t cbCount = 0;
volatile uint8_t lastStatus = 255;
volatile uint16_t lastDist = 0xFFFF;
volatile uint16_t d1_cm = 0xFFFF;

UWBRangingController *controller = nullptr;

void sendTelemetry(uint32_t cb, uint8_t status, uint16_t lastDistance, uint16_t d1, uint32_t nowMs) {
  Serial1.print("UWB");
  Serial1.print(",session=0x");
  Serial1.print(THIS_SESSION_ID, HEX);
  Serial1.print(",src=0x");
  Serial1.print(SRC_ADDR[0], HEX);
  Serial1.print(SRC_ADDR[1], HEX);
  Serial1.print(",dst=0x");
  Serial1.print(DST_ADDR[0], HEX);
  Serial1.print(DST_ADDR[1], HEX);
  Serial1.print(",cb=");
  Serial1.print(cb);
  Serial1.print(",status=");
  Serial1.print(status);
  Serial1.print(",lastDist=");
  Serial1.print(lastDistance);
  Serial1.print(",d1=");
  if (d1 == 0xFFFF) {
    Serial1.print(-1);
  } else {
    Serial1.print(d1);
  }
  Serial1.print(",ms=");
  Serial1.println(nowMs);
}

void rangingHandler(UWBRangingData &data) {
  if (data.measureType() != (uint8_t)uwb::MeasurementType::TWO_WAY) return;

  RangingMeasures m = data.twoWayRangingMeasure();
  for (int i = 0; i < data.available(); i++) {
    cbCount++;
    lastStatus = m[i].status;
    lastDist = m[i].distance;

    if (m[i].status == 0 && m[i].distance != 0xFFFF) {
      d1_cm = m[i].distance;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial1.begin(UART_BAUD);

  Serial.println("PORTENTA CONTINUOUS CONTROLLER");
  Serial.print("UART1 telemetry baud=");
  Serial.println(UART_BAUD);

  UWB.registerRangingCallback(rangingHandler);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  controller = new UWBRangingController(THIS_SESSION_ID, src, dst);
  controller->appParams.slotPerRR(8);
  controller->appParams.slotDuration(1200);
  controller->appParams.rangingDuration(20);
  controller->appParams.maxRetries(0);
  controller->init();
  controller->start();

  Serial.print("session=0x");
  Serial.println(THIS_SESSION_ID, HEX);
}

void loop() {
  static uint32_t t0 = 0;
  uint32_t now = millis();

  if (now - t0 >= PRINT_MS) {
    t0 += PRINT_MS;

    uint32_t c = cbCount;
    uint8_t s = lastStatus;
    uint16_t ld = lastDist;
    uint16_t d1 = d1_cm;

    Serial.print("cb=");
    Serial.print(c);
    Serial.print(" lastStatus=");
    Serial.print(s);
    Serial.print(" lastDist=");
    Serial.print(ld);
    Serial.print(" D1=");
    if (d1 == 0xFFFF) Serial.print("--");
    else Serial.print(d1);
    Serial.println(" cm");

    sendTelemetry(c, s, ld, d1, now);
  }

  delay(1);
}

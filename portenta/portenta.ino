#include <PortentaUWBShield.h>

uint8_t SRC_ADDR[] = {0x11, 0x11};   // Portenta
uint8_t DST_ADDR[] = {0x22, 0x22};   // Stella

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("PORTENTA ANCHOR");

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBRangingControlee anchor(0x12345678, src, dst);
  UWBSessionManager.addSession(anchor);

  anchor.init();
  anchor.start();

  Serial.println("Anchor pret");
}

void loop() {
  delay(10);
}

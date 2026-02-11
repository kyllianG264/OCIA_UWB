#include <StellaUWB.h>

uint8_t SRC_ADDR[] = {0x22, 0x22};   // Stella
uint8_t DST_ADDR[] = {0x11, 0x11};   // Anchor Master

void setup() {
  Serial.begin(115200);
  delay(200); // sur pile: ne pas bloquer

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  // Stella en mode RESPONDER / CONTROLEE
  UWBTracker tag(0x12345678, src, dst,
                 uwb::DeviceRole::RESPONDER,
                 uwb::DeviceType::CONTROLEE);

  UWBSessionManager.addSession(tag);

  tag.init();
  tag.start();
}

void loop() {
  delay(10);
}

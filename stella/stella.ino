#include <StellaUWB.h>

uint8_t SRC_ADDR[] = {0x22, 0x22};   // Stella
uint8_t DST_ADDR[] = {0x11, 0x11};   // Portenta

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("STELLA RESPONDER");

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBTracker tag(0x12345678, src, dst);
  UWBSessionManager.addSession(tag);

  tag.init();
  tag.start();

  Serial.println("Stella prete (pas d'affichage distance)");
}

void loop() {
  delay(10);
}

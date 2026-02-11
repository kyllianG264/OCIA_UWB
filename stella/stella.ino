#include <StellaUWB.h>

uint8_t SRC_ADDR[] = {0x22, 0x22};   // Stella
uint8_t DST1_ADDR[] = {0x11, 0x11};  // Anchor1
uint8_t DST2_ADDR[] = {0x33, 0x33};  // Anchor2

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("STELLA RESPONDER (silencieuse)");

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst1(UWBMacAddress::Size::SHORT, DST1_ADDR);
  UWBMacAddress dst2(UWBMacAddress::Size::SHORT, DST2_ADDR);

  // 2 sessions pour répondre aux 2 ancres
  UWBTracker resp1(0x12345678, src, dst1);
  UWBTracker resp2(0x12345679, src, dst2);

  UWBSessionManager.addSession(resp1);
  UWBSessionManager.addSession(resp2);

  resp1.init(); resp2.init();
  resp1.start(); resp2.start();

  Serial.println("Stella prete (responder)");
}

void loop() {
  delay(10);
}

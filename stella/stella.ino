#include <StellaUWB.h>

// Stella tag/responder address and Portenta anchor/controller address.
uint8_t SRC_ADDR[] = {0x22, 0x22};
uint8_t DST_ADDR[] = {0x11, 0x11};

static const uint32_t SESSION_ID = 0x12345678;

void setup()
{
  Serial.begin(115200);
  delay(200);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
  UWBMacAddress dst(UWBMacAddress::Size::SHORT, DST_ADDR);

  UWBTracker tag(SESSION_ID, src, dst,
                 uwb::DeviceRole::RESPONDER,
                 uwb::DeviceType::CONTROLEE);

  UWBSessionManager.addSession(tag);

  tag.init();
  tag.start();
}

void loop()
{
  delay(10);
}

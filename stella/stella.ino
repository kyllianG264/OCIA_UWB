#include <StellaUWB.h>

// Stella rotates across the configured sessions fast enough to give each
// Portenta a 10 Hz response opportunity.
//
// If ACTIVE_PORTENTA_COUNT = 2, Stella alternates over 2 sessions.
// If ACTIVE_PORTENTA_COUNT = 3, Stella alternates over 3 sessions.
// To keep 10 Hz per Portenta, the total rotation rate increases with the count.

static const uint32_t TARGET_HZ_PER_PORTENTA = 10;
static const uint32_t SLOT_GUARD_MS = 3;
static const size_t MAX_PORTENTA_COUNT = 4;
static const size_t ACTIVE_PORTENTA_COUNT = 3;

static_assert(ACTIVE_PORTENTA_COUNT > 0, "Need at least one active Portenta");
static_assert(ACTIVE_PORTENTA_COUNT <= MAX_PORTENTA_COUNT, "Active count exceeds max count");

static uint8_t STELLA_ADDR[] = {0x22, 0x22};

static uint8_t PORTENTA_ADDRS[MAX_PORTENTA_COUNT][2] = {
  {0x11, 0x11},
  {0x33, 0x33},
  {0x44, 0x44},
  {0x55, 0x55},
};

static const uint32_t SESSION_IDS[MAX_PORTENTA_COUNT] = {
  0x1001,
  0x1002,
  0x1003,
  0x1004,
};

static const uint32_t SUPERFRAME_MS = 1000 / TARGET_HZ_PER_PORTENTA;
static const uint32_t WINDOW_MS = (SUPERFRAME_MS / ACTIVE_PORTENTA_COUNT) - SLOT_GUARD_MS;

UWBRangingControlee *sessions[MAX_PORTENTA_COUNT];
size_t activeIndex = MAX_PORTENTA_COUNT;

size_t currentSlotIndex(uint32_t nowMs) {
  return (size_t)(((uint64_t)(nowMs % SUPERFRAME_MS) * ACTIVE_PORTENTA_COUNT) / SUPERFRAME_MS);
}

void activateSlot(size_t index) {
  if (index == activeIndex) return;

  if (activeIndex < ACTIVE_PORTENTA_COUNT) {
    sessions[activeIndex]->stop();
  }

  sessions[index]->start();
  activeIndex = index;
}

void setup() {
  Serial.begin(115200);
  delay(200);

  UWB.begin();
  while (UWB.state() != 0) delay(10);

  for (size_t i = 0; i < ACTIVE_PORTENTA_COUNT; ++i) {
    UWBMacAddress src(UWBMacAddress::Size::SHORT, STELLA_ADDR);
    UWBMacAddress dst(UWBMacAddress::Size::SHORT, PORTENTA_ADDRS[i]);

    sessions[i] = new UWBRangingControlee(SESSION_IDS[i], src, dst);
    sessions[i]->appParams.slotPerRR(8);
    sessions[i]->appParams.slotDuration(1200);
    sessions[i]->appParams.rangingDuration(WINDOW_MS);
    sessions[i]->appParams.maxRetries(0);
    sessions[i]->init();
  }

  activateSlot(0);

  Serial.print("Stella slicer ready active=");
  Serial.print(ACTIVE_PORTENTA_COUNT);
  Serial.print(" superframe=");
  Serial.print(SUPERFRAME_MS);
  Serial.print("ms window=");
  Serial.print(WINDOW_MS);
  Serial.println("ms");
}

void loop() {
  activateSlot(currentSlotIndex(millis()));
  delay(1);
}

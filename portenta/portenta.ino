#include "UWBFeature.h"

#define ANCHOR_ID 1

UWBFeature uwbFeature;

void setup()
{
    Serial.begin(115200);   // debug USB
    Serial1.begin(115200);  // UART vers ESP
    delay(2000);

    Serial.println("=== PORTENTA START ===");
    Serial1.println("BOOT Portenta UWB anchor");

    uwbFeature.begin();

    Serial.println("=== PORTENTA READY ===");
    Serial1.println("READY Portenta UWB anchor");
}

void loop()
{
    static uint32_t seq = 0;
    char msg[96];

    uwbFeature.update();

    if (uwbFeature.hasValidDistance())
    {
        snprintf(
            msg,
            sizeof(msg),
            "A%d=%u A=%d S=%lu D1=%u status=%u cb=%lu",
            ANCHOR_ID,
            uwbFeature.getDistanceCm(),
            ANCHOR_ID,
            (unsigned long)seq++,
            uwbFeature.getDistanceCm(),
            uwbFeature.getLastStatus(),
            (unsigned long)uwbFeature.getCallbackCount()
        );
    }
    else
    {
        snprintf(
            msg,
            sizeof(msg),
            "A%d=-- A=%d S=%lu D1=-- status=%u cb=%lu",
            ANCHOR_ID,
            ANCHOR_ID,
            (unsigned long)seq++,
            uwbFeature.getLastStatus(),
            (unsigned long)uwbFeature.getCallbackCount()
        );
    }

    Serial.println(msg);
    Serial1.println(msg);

    delay(100);
}

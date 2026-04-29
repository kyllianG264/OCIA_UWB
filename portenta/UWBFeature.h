#ifndef UWB_FEATURE_H
#define UWB_FEATURE_H

#include <Arduino.h>
#include <PortentaUWBShield.h>

class UWBFeature
{
public:
    UWBFeature();

    void begin();
    void update();

    bool hasValidDistance() const;
    uint16_t getDistanceCm() const;
    uint16_t getLastRawDistanceCm() const;
    uint8_t getLastStatus() const;
    uint32_t getCallbackCount() const;

private:
    static void rangingHandler(UWBRangingData &data);

    static volatile uint16_t _distanceCm;
    static volatile uint16_t _lastRawDistanceCm;
    static volatile uint8_t _lastStatus;
    static volatile uint32_t _callbackCount;
    static bool _callbackRegistered;

    UWBRangingController* _controller;
};

#endif

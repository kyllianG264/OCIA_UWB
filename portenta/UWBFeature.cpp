#include "UWBFeature.h"

static uint8_t SRC_ADDR[] = {0x11, 0x11};
static uint8_t TAG_ADDR[] = {0x22, 0x22};
static const uint32_t SESSION_ID = 0x12345678;

volatile uint16_t UWBFeature::_distanceCm = 0xFFFF;
volatile uint16_t UWBFeature::_lastRawDistanceCm = 0xFFFF;
volatile uint8_t UWBFeature::_lastStatus = 255;
volatile uint32_t UWBFeature::_callbackCount = 0;
bool UWBFeature::_callbackRegistered = false;

UWBFeature::UWBFeature() : _controller(nullptr)
{
}

void UWBFeature::rangingHandler(UWBRangingData &data)
{
    if (data.measureType() != (uint8_t)uwb::MeasurementType::TWO_WAY)
        return;

    RangingMeasures m = data.twoWayRangingMeasure();

    for (int i = 0; i < data.available(); i++)
    {
        _callbackCount++;
        _lastStatus = m[i].status;
        _lastRawDistanceCm = m[i].distance;

        if (m[i].status == 0 && m[i].distance != 0xFFFF)
        {
            _distanceCm = m[i].distance;
        }
    }
}

void UWBFeature::begin()
{
    Serial.println("UWB INIT");

    _distanceCm = 0xFFFF;
    _lastRawDistanceCm = 0xFFFF;
    _lastStatus = 255;
    _callbackCount = 0;

    if (!_callbackRegistered)
    {
        UWB.registerRangingCallback(UWBFeature::rangingHandler);
        _callbackRegistered = true;
    }

    UWB.begin();

    while (UWB.state() != 0)
        delay(10);

    UWBMacAddress src(UWBMacAddress::Size::SHORT, SRC_ADDR);
    UWBMacAddress tag(UWBMacAddress::Size::SHORT, TAG_ADDR);

    _controller = new UWBRangingController(SESSION_ID, src, tag);
    UWBSessionManager.addSession(*_controller);

    _controller->init();
    _controller->start();

    Serial.println("UWB READY");
}

void UWBFeature::update()
{
}

bool UWBFeature::hasValidDistance() const
{
    return _distanceCm != 0xFFFF;
}

uint16_t UWBFeature::getDistanceCm() const
{
    return _distanceCm;
}

uint16_t UWBFeature::getLastRawDistanceCm() const
{
    return _lastRawDistanceCm;
}

uint8_t UWBFeature::getLastStatus() const
{
    return _lastStatus;
}

uint32_t UWBFeature::getCallbackCount() const
{
    return _callbackCount;
}

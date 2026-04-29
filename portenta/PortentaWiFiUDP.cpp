#include "PortentaWiFiUDP.h"

PortentaWiFiUDP::PortentaWiFiUDP()
    : _ssid(nullptr),
      _pass(nullptr),
      _destIp(0, 0, 0, 0),
      _destPort(0),
      _localPort(0),
      _useBroadcast(false)
{
}

void PortentaWiFiUDP::begin(const char* ssid,
                            const char* pass,
                            IPAddress destIp,
                            uint16_t destPort,
                            uint16_t localPort)
{
    _ssid = ssid;
    _pass = pass;
    _destIp = destIp;
    _destPort = destPort;
    _localPort = localPort;
    _useBroadcast = false;

    connect();
}

void PortentaWiFiUDP::beginBroadcast(const char* ssid,
                                     const char* pass,
                                     uint16_t destPort,
                                     uint16_t localPort)
{
    _ssid = ssid;
    _pass = pass;
    _destPort = destPort;
    _localPort = localPort;
    _useBroadcast = true;

    connect();
    _destIp = broadcastAddress();
}

void PortentaWiFiUDP::connect()
{
    Serial.println("[WIFI] start");

    WiFi.begin(_ssid, _pass);

    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");

        if (millis() - start > 20000)
        {
            Serial.println("\n[WIFI] retry");
            WiFi.disconnect();
            WiFi.begin(_ssid, _pass);
            start = millis();
        }
    }

    Serial.println();
    Serial.println("[WIFI] connected");
    Serial.println(WiFi.localIP());

    _udp.begin(_localPort);

    Serial.println("[UDP] ready");
}

bool PortentaWiFiUDP::connected()
{
    return WiFi.status() == WL_CONNECTED;
}

IPAddress PortentaWiFiUDP::broadcastAddress()
{
    IPAddress ip = WiFi.localIP();
    IPAddress mask = WiFi.subnetMask();

    if (ip == IPAddress(0, 0, 0, 0))
    {
        return IPAddress(255, 255, 255, 255);
    }

    return IPAddress(
        ip[0] | (uint8_t)(~mask[0]),
        ip[1] | (uint8_t)(~mask[1]),
        ip[2] | (uint8_t)(~mask[2]),
        ip[3] | (uint8_t)(~mask[3])
    );
}

void PortentaWiFiUDP::send(const char* msg)
{
    if (!connected())
    {
        connect();
    }

    if (_useBroadcast)
    {
        _destIp = broadcastAddress();
    }

    _udp.beginPacket(_destIp, _destPort);
    _udp.write((const uint8_t*)msg, strlen(msg));
    _udp.endPacket();
}

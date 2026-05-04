#ifndef PORTENTA_WIFI_UDP_H
#define PORTENTA_WIFI_UDP_H

#include <Arduino.h>
#if __has_include(<WiFiC3.h>)
#include <WiFiC3.h>
#elif __has_include(<WiFi.h>)
#include <WiFi.h>
#else
#error "WiFi library not found. Install the WiFi library for your Portenta board."
#endif
#include <WiFiUdp.h>

class PortentaWiFiUDP
{
public:
    PortentaWiFiUDP();

    void begin(const char* ssid,
               const char* pass,
               IPAddress destIp,
               uint16_t destPort,
               uint16_t localPort);

    void beginBroadcast(const char* ssid,
                        const char* pass,
                        uint16_t destPort,
                        uint16_t localPort);

    bool connected();
    IPAddress broadcastAddress();
    void send(const char* msg);

private:
    const char* _ssid;
    const char* _pass;
    IPAddress _destIp;
    uint16_t _destPort;
    uint16_t _localPort;
    bool _useBroadcast;
    WiFiUDP _udp;

    void connect();
};

#endif

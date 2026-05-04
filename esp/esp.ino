#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

const char* WIFI_SSID = "HONOR Magic6 Lite 5G";
const char* WIFI_PASSWORD = "kyllian264";

const unsigned int DEST_PORT = 4210;
const unsigned int LOCAL_PORT = 5001;
const unsigned long NO_DATA_TIMEOUT_MS = 1000;
const unsigned long WIFI_RETRY_MS = 2000;

WiFiUDP udp;
String line = "";

unsigned long lastSend = 0;
unsigned long lastWifiTry = 0;

IPAddress getBroadcastAddress()
{
    IPAddress ip = WiFi.localIP();
    IPAddress mask = WiFi.subnetMask();

    if (ip == IPAddress(0, 0, 0, 0)) {
        return IPAddress(255, 255, 255, 255);
    }

    return IPAddress(
        ip[0] | (uint8_t)(~mask[0]),
        ip[1] | (uint8_t)(~mask[1]),
        ip[2] | (uint8_t)(~mask[2]),
        ip[3] | (uint8_t)(~mask[3])
    );
}

void connectWifi()
{
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }
}

void sendUdpLine(const String &payload)
{
    if (WiFi.status() != WL_CONNECTED) {
        return;
    }

    IPAddress broadcastIp = getBroadcastAddress();
    udp.beginPacket(broadcastIp, DEST_PORT);
    udp.print(payload);
    udp.endPacket();
}

void setup()
{
    Serial.begin(115200);
    line.reserve(160);

    delay(1000);
    connectWifi();

    udp.begin(LOCAL_PORT);
    lastSend = millis();

    String bootLine = "ESP_READY broadcast=";
    bootLine += getBroadcastAddress().toString();
    sendUdpLine(bootLine);
}

void loop()
{
    if (WiFi.status() != WL_CONNECTED && millis() - lastWifiTry > WIFI_RETRY_MS) {
        lastWifiTry = millis();
        connectWifi();
    }

    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n') {
            if (line.length() > 0) {
                sendUdpLine(line);
            }

            line = "";
            lastSend = millis();
        } else if (c != '\r') {
            line += c;
        }
    }

    if (millis() - lastSend > NO_DATA_TIMEOUT_MS) {
        sendUdpLine("NO_DATA");
        lastSend = millis();
    }
}

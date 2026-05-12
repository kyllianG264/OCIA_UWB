#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "secrets.h"

// ESP32 bridge for Portenta UWB telemetry.
// Hardware UART mapping follows PCB/ESP32_PIN_MAPPING.md:
// - ESP32 GPIO16 / U2RXD <= PORTENTA_TX
// - ESP32 GPIO17 / U2TXD => PORTENTA_RX

static const uint32_t DEBUG_BAUD = 115200;
static const uint32_t PORTENTA_UART_BAUD = 115200;
static const int PORTENTA_RX_PIN = 16;
static const int PORTENTA_TX_PIN = 17;
static const size_t LINE_BUFFER_SIZE = 160;
static const uint32_t STALE_TIMEOUT_MS = 1500;
static const uint32_t WIFI_RETRY_MS = 5000;

#ifndef WIFI_UDP_TARGET_PORT
#define WIFI_UDP_TARGET_PORT 4210
#endif

struct UwbTelemetry {
  String session;
  String src;
  String dst;
  uint32_t callbackCount = 0;
  int status = -1;
  int lastDistanceCm = -1;
  int d1Cm = -1;
  uint32_t portentaMillis = 0;
  uint32_t receivedAt = 0;
  bool valid = false;
};

HardwareSerial PortentaSerial(2);
WiFiUDP Udp;
char lineBuffer[LINE_BUFFER_SIZE];
size_t lineLength = 0;
UwbTelemetry lastTelemetry;
IPAddress udpTargetIp;
bool udpTargetConfigured = false;

bool parseIpAddress(const char *text, IPAddress &outIp) {
  return outIp.fromString(text) == 1;
}

IPAddress subnetBroadcast(IPAddress ip, IPAddress subnet) {
  IPAddress broadcast;
  for (int i = 0; i < 4; ++i) {
    broadcast[i] = (uint8_t)(ip[i] | ~subnet[i]);
  }
  return broadcast;
}

void configureUdpTarget() {
#ifdef WIFI_UDP_TARGET_IP
  if (parseIpAddress(WIFI_UDP_TARGET_IP, udpTargetIp)) {
    udpTargetConfigured = true;
    Serial.print("UDP target IP from secrets.h: ");
    Serial.println(udpTargetIp);
    return;
  }

  Serial.println("Invalid WIFI_UDP_TARGET_IP, falling back to broadcast");
#endif

  udpTargetIp = subnetBroadcast(WiFi.localIP(), WiFi.subnetMask());
  udpTargetConfigured = true;
  Serial.print("UDP target IP broadcast: ");
  Serial.println(udpTargetIp);
}

void connectWifiIfNeeded() {
  static uint32_t lastAttemptMs = 0;

  if (WiFi.status() == WL_CONNECTED) return;

  uint32_t now = millis();
  if (now - lastAttemptMs < WIFI_RETRY_MS) return;
  lastAttemptMs = now;

  Serial.print("Connecting WiFi SSID=");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t startMs = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startMs) < 10000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected IP=");
    Serial.println(WiFi.localIP());
    configureUdpTarget();
  } else {
    Serial.println("WiFi connect timeout");
  }
}

void sendUdpLine(const String &line) {
  if (WiFi.status() != WL_CONNECTED) return;
  if (!udpTargetConfigured) configureUdpTarget();

  Udp.beginPacket(udpTargetIp, WIFI_UDP_TARGET_PORT);
  Udp.print(line);
  Udp.endPacket();
}

bool parseUnsignedField(const String &line, const char *key, uint32_t &outValue) {
  String needle = String(",") + key + "=";
  int start = line.indexOf(needle);
  if (start < 0) return false;
  start += needle.length();
  int end = line.indexOf(',', start);
  String value = (end < 0) ? line.substring(start) : line.substring(start, end);
  outValue = (uint32_t)value.toInt();
  return value.length() > 0;
}

bool parseSignedField(const String &line, const char *key, int &outValue) {
  String needle = String(",") + key + "=";
  int start = line.indexOf(needle);
  if (start < 0) return false;
  start += needle.length();
  int end = line.indexOf(',', start);
  String value = (end < 0) ? line.substring(start) : line.substring(start, end);
  outValue = value.toInt();
  return value.length() > 0;
}

bool parseStringField(const String &line, const char *key, String &outValue) {
  String needle = String(",") + key + "=";
  int start = line.indexOf(needle);
  if (start < 0) return false;
  start += needle.length();
  int end = line.indexOf(',', start);
  outValue = (end < 0) ? line.substring(start) : line.substring(start, end);
  return outValue.length() > 0;
}

bool parseTelemetryLine(const String &line, UwbTelemetry &outTelemetry) {
  if (!line.startsWith("UWB,")) return false;

  UwbTelemetry parsed;
  if (!parseStringField(line, "session", parsed.session)) return false;
  if (!parseStringField(line, "src", parsed.src)) return false;
  if (!parseStringField(line, "dst", parsed.dst)) return false;
  if (!parseUnsignedField(line, "cb", parsed.callbackCount)) return false;
  if (!parseSignedField(line, "status", parsed.status)) return false;
  if (!parseSignedField(line, "lastDist", parsed.lastDistanceCm)) return false;
  if (!parseSignedField(line, "d1", parsed.d1Cm)) return false;
  if (!parseUnsignedField(line, "ms", parsed.portentaMillis)) return false;

  parsed.receivedAt = millis();
  parsed.valid = true;
  outTelemetry = parsed;
  return true;
}

void printTelemetry(const UwbTelemetry &telemetry) {
  Serial.print("UWB RX session=");
  Serial.print(telemetry.session);
  Serial.print(" src=");
  Serial.print(telemetry.src);
  Serial.print(" dst=");
  Serial.print(telemetry.dst);
  Serial.print(" cb=");
  Serial.print(telemetry.callbackCount);
  Serial.print(" status=");
  Serial.print(telemetry.status);
  Serial.print(" lastDist=");
  Serial.print(telemetry.lastDistanceCm);
  Serial.print(" d1=");
  if (telemetry.d1Cm < 0) Serial.print("--");
  else Serial.print(telemetry.d1Cm);
  Serial.print(" cm age=");
  Serial.print(millis() - telemetry.receivedAt);
  Serial.println(" ms");
}

void handleLine(const char *rawLine) {
  String line(rawLine);
  line.trim();
  if (line.length() == 0) return;

  sendUdpLine(line);

  UwbTelemetry parsed;
  if (!parseTelemetryLine(line, parsed)) {
    Serial.print("Ignored line: ");
    Serial.println(line);
    return;
  }

  lastTelemetry = parsed;
  printTelemetry(lastTelemetry);
}

void pollPortentaUart() {
  while (PortentaSerial.available() > 0) {
    char c = (char)PortentaSerial.read();
    if (c == '\r') continue;

    if (c == '\n') {
      lineBuffer[lineLength] = '\0';
      handleLine(lineBuffer);
      lineLength = 0;
      continue;
    }

    if (lineLength < (LINE_BUFFER_SIZE - 1)) {
      lineBuffer[lineLength++] = c;
    } else {
      lineLength = 0;
      Serial.println("UART line overflow, buffer reset");
    }
  }
}

void printHeartbeat() {
  static uint32_t lastHeartbeatMs = 0;
  uint32_t now = millis();
  if (now - lastHeartbeatMs < 1000) return;
  lastHeartbeatMs = now;

  Serial.print("ESP bridge alive");
  Serial.print(" wifi=");
  Serial.print(WiFi.status() == WL_CONNECTED ? "up" : "down");
  if (lastTelemetry.valid) {
    Serial.print(" lastSession=");
    Serial.print(lastTelemetry.session);
    Serial.print(" lastAge=");
    Serial.print(now - lastTelemetry.receivedAt);
    Serial.print(" ms");
    if (now - lastTelemetry.receivedAt > STALE_TIMEOUT_MS) {
      Serial.print(" stale=yes");
    }
  } else {
    Serial.print(" waiting_for_uwb=yes");
  }
  Serial.println();
}

void setup() {
  Serial.begin(DEBUG_BAUD);
  delay(200);

  PortentaSerial.begin(PORTENTA_UART_BAUD, SERIAL_8N1, PORTENTA_RX_PIN, PORTENTA_TX_PIN);

  Serial.println("ESP32 Portenta UWB bridge");
  Serial.print("UART2 baud=");
  Serial.print(PORTENTA_UART_BAUD);
  Serial.print(" RX=");
  Serial.print(PORTENTA_RX_PIN);
  Serial.print(" TX=");
  Serial.println(PORTENTA_TX_PIN);
  Serial.print("UDP target port=");
  Serial.println(WIFI_UDP_TARGET_PORT);

  connectWifiIfNeeded();
}

void loop() {
  connectWifiIfNeeded();
  pollPortentaUart();
  printHeartbeat();
  delay(2);
}

# Librairies Arduino compatibles

Les archives compatibles sont deja dans le repo :

- `ArduinoCore-mbed-main.zip`
- `PortentaUWBShield-main.zip`
- `StellaUWB-main.zip`

Installation Arduino IDE :

1. Installer le core mbed pour Portenta depuis `ArduinoCore-mbed-main.zip` si ton IDE ne l'a pas deja.
2. Installer `PortentaUWBShield-main.zip` via `Croquis > Inclure une bibliotheque > Ajouter la bibliotheque .ZIP...`.
3. Installer `StellaUWB-main.zip` de la meme maniere.

Les sketches du repo utilisent :

- `#include <PortentaUWBShield.h>` pour `portenta/portenta.ino`
- `#include <StellaUWB.h>` pour `stella/stella.ino`
- `#include <ESP8266WiFi.h>` et `#include <WiFiUdp.h>` pour `esp/esp.ino`

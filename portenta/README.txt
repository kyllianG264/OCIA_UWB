Arduino IDE :

Ouvrir directement :
portenta/portenta.ino

Le dossier est un sketch Arduino propre :
- portenta.ino          : programme principal
- UWBFeature.h/.cpp     : feature UWB
- PortentaWiFiUDP.h/.cpp: feature Wi-Fi UDP optionnelle
- testUWB.txt           : sketch de test UWB a copier dans portenta.ino si besoin
- testWIFI.txt          : sketch de test Wi-Fi UDP a copier dans portenta.ino si besoin

Flux principal actuel :
Portenta UWB -> Serial1 -> ESP8266 -> UDP broadcast -> Python

Format serie/UDP :
A1=123 A=1 S=0 D1=123 status=0 cb=42

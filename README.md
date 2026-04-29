# OCIA UWB

Depot organise pour une chaine UWB complete :

```text
Portenta UWB anchor -> UART Serial1 -> ESP8266 UDP broadcast -> Python receiver
Stella UWB tag      -> UWB responder
```

## Arduino

- `portenta/portenta.ino` : Portenta en anchor/controller. Il imprime `A1=<cm> D1=<cm> ...` en USB et sur `Serial1`.
- `portenta/UWBFeature.*` : feature UWB separee, reprise de ton decoupage.
- `portenta/PortentaWiFiUDP.*` : feature Wi-Fi/UDP optionnelle pour envoyer depuis la Portenta directement.
- `stella/stella.ino` : Stella en tag/responder.
- `esp/esp.ino` : bridge ESP8266. Il lit les lignes serie et les envoie en UDP broadcast sur le port `4210`.
- `libraries/README.md` : archives de librairies compatibles a installer dans Arduino IDE.

Pour plusieurs ancres, dupliquer le sketch Portenta et changer :

- `ANCHOR_ID`
- `SRC_ADDR`
- si besoin la session/adresse selon ton montage

Le Python sait lire `A1=123`, `A2=456`, etc.

## Python

Receiver simple :

```powershell
py python\udp_distance_viewer.py --port 4210 --anchor 1
```

Estimations :

```powershell
py -m pip install -r python\requirements.txt
py python\estimation_3d\main.py
py python\estimation_2d\main.py
py python\estimation_3d_to_2d\main.py
```

Mode UDP pour les estimateurs 2D :

```powershell
py python\estimation_2d\main.py --source udp --port 4210
py python\estimation_3d_to_2d\main.py --source udp --port 4210
```

## Cabling rapide

- Portenta `Serial1 TX` vers ESP8266 `RX`
- masses communes entre Portenta et ESP
- meme baudrate : `115200`
- PC et ESP sur le meme reseau Wi-Fi

L'ESP calcule l'adresse broadcast du reseau avec `WiFi.localIP()` et `WiFi.subnetMask()` puis envoie vers le port `4210`.

# ESP32 Header Mapping

This project currently uses generic 1x15 connector symbols for the ESP32 dev board
headers in the KiCad schematic, not a custom ESP32 symbol with named GPIO pins.

That means the electrical mapping is valid, but the visible pin names remain generic
(`Pin_1_1`, `Pin_2_2`, etc.) until a dedicated symbol is created.

## ESP32 Right Header Physical Order

Based on the local ESP32 documentation images in `DOC TECHNIQUE/ESP/`, the right-side
header of the ESP32 dev board maps as follows:

| Header Pin | ESP32 Signal |
| --- | --- |
| 1 | 3V3 |
| 2 | GND |
| 3 | GPIO15 |
| 4 | GPIO2 |
| 5 | GPIO4 |
| 6 | GPIO16 / U2RXD |
| 7 | GPIO17 / U2TXD |
| 8 | GPIO5 |
| 9 | GPIO18 / SCK |
| 10 | GPIO19 / MISO |
| 11 | GPIO21 |
| 12 | GPIO3 / U0RXD |
| 13 | GPIO1 / U0TXD |
| 14 | GPIO22 |
| 15 | GPIO23 / MOSI |

## Current Schematic Mapping

Current active mapping on `J_ESP32_LEFT` in the schematic:

| Header Pin | ESP32 GPIO | Net |
| --- | --- | --- |
| 1 | 3V3 | `+3V3` |
| 2 | GND | `GND` |
| 3 | GPIO15 | `W5500_INT` |
| 4 | GPIO2 | `SPI_CS` |
| 5 | GPIO4 | not connected |
| 6 | GPIO16 / U2RXD | `PORTENTA_TX` |
| 7 | GPIO17 / U2TXD | `PORTENTA_RX` |
| 8 | GPIO5 | `W5500_RST` |
| 9 | GPIO18 | `SPI_SCK` |
| 10 | GPIO19 | `SPI_MISO` |
| 11 | GPIO21 | not connected |
| 12 | GPIO3 / U0RXD | not connected |
| 13 | GPIO1 / U0TXD | not connected |
| 14 | GPIO22 | not connected |
| 15 | GPIO23 | `SPI_MOSI` |

## Important Note

The PCB file may still reflect an older routing state until KiCad runs:

`Tools -> Update PCB from Schematic...`

After updating, verify the rerouted ESP32 pins before manufacturing.

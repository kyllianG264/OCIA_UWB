# Python UWB tools

## Receiver UDP simple

```powershell
py python\udp_distance_viewer.py --port 4210 --anchor 1
```

Le viewer ecoute les broadcasts UDP de l'ESP et accepte ces formats :

- `D1=123`
- `A1=123`
- `anchor=1,distance_cm=123`

## Simulations / estimations

Chaque estimation garde la meme architecture :

- `scene.py` : monde, ancres, reglages, cache
- `real_tag.py` : mouvement simule du tag
- `uwb_sources.py` : distances simulees, avec UDP active pour les apps 2D
- `position_calcul.py` : solveur, lissage, metriques
- `display.py` : rendu
- `main.py` : boucle principale

Lancements :

```powershell
py -m pip install -r python\requirements.txt
py python\estimation_3d\main.py
py python\estimation_2d\main.py
py python\estimation_3d_to_2d\main.py
```

Les apps 2D et 3D-vers-2D acceptent aussi le flux ESP :

```powershell
py python\estimation_2d\main.py --source udp --port 4210
py python\estimation_3d_to_2d\main.py --source udp --port 4210
```

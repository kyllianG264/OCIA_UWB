Architecture Estimation 2D :

- scene.py            -> monde 2D, ancres, constantes, cache
- real_tag.py         -> mouvement simule du tag
- uwb_sources.py      -> distances simulees ou UDP
- position_calcul.py  -> solveur 2D + lissages + metriques
- display.py          -> rendu 2D + HUD
- main.py             -> boucle principale

Lancement simulation :
py python\estimation_2d\main.py

Lancement avec ESP UDP :
py python\estimation_2d\main.py --source udp --port 4210

Format attendu en UDP :
A1=123
A2=456
ou anchor=1,distance_cm=123

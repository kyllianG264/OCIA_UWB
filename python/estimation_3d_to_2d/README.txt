Architecture Estimation 3D -> 2D :

- scene.py            -> monde 2D, hauteurs, ancres, constantes, cache
- real_tag.py         -> mouvement simule du tag + sauts verticaux
- uwb_sources.py      -> distances UWB 3D simulees ou UDP, puis projection au sol
- position_calcul.py  -> solveur 2D + lissages + metriques
- display.py          -> rendu 2D + HUD
- main.py             -> boucle principale

Lancement simulation :
py python\estimation_3d_to_2d\main.py

Lancement avec ESP UDP :
py python\estimation_3d_to_2d\main.py --source udp --port 4210

Format attendu en UDP :
A1=123
A2=456
ou anchor=1,distance_cm=123

Dans ce mode, les distances recues sont considerees comme distances UWB 3D.
Elles sont converties en rayons 2D avec :
sqrt(distance_3d^2 - (hauteur_ancre - hauteur_tag_supposee)^2)

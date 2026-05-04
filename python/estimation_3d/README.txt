Architecture Estimation 3D :

- scene.py            -> monde 3D, ancres, constantes, reset
- real_tag.py         -> mouvement simule du tag
- uwb_sources.py      -> recuperation des distances UWB simulees
- position_calcul.py  -> solveur 3D + lissages + metriques
- display.py          -> rendu 3D + HUD
- main.py             -> boucle principale

Lancement :
py python\estimation_3d\main.py

Controles :
- H : replier/deplier le HUD
- Fleches haut/bas : selectionner un parametre editable
- Fleches gauche/droite : modifier la valeur en temps reel
- Shift + gauche/droite : pas plus grand
- 4 / 5 / 6 : changer le nombre d'ancres actives

Cache :
- les reglages de lissage, de tolerance et les positions des ancres sont sauves dans settings_cache.json
- le bruit UWB est aussi editable en direct et sauvegarde

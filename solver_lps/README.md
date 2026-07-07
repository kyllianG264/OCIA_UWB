# solver_lps

Ce dossier contient le solver Python final du projet.

## Role

- lancer l'application
- centraliser les assets utilises par le solver
- regrouper les briques metier CV, UWB, terrain et joueurs
- contenir la couche presentation qui affiche les pages et orchestre la navigation

## Entree principale

- `main.py` : point d'entree principal du solver
- `bootstrap_solver.py` : aide au lancement et a la preparation de l'environnement
- `launch_solver.bat` : lancement Windows

## Sous-dossiers principaux

- `assets` : donnees, videos, sorties et modeles utilises par le solver
- `features` : logique metier du solver
- `presentation` : pages UI, navigation et affichage

## Regle simple

Le code final doit vivre ici. Les dossiers generes automatiquement comme `__pycache__` et l'environnement `.solver_env` ne font pas partie de l'architecture metier.

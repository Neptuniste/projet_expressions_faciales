# Reconnaissance d'expressions faciales — projet Deep Learning

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `Projet_expressions_faciales.ipynb` | notebook principal, parties 1 à 10 du sujet, exécuté (avec toutes les sorties) |
| `Red_team_expressions_faciales.ipynb` | notebook compagnon : on essaie de mettre le modèle en défaut (enrichissement) |
| `demo_webcam.py` | démonstration en direct : webcam → YOLO → CNN → expression |
| `installer_demo.bat`, `lancer_demo.bat` | installation et lancement de la démo sous Windows |
| `requirements_demo.txt` | bibliothèques nécessaires à la démo |
| `modeles/modele_final.keras` | le CNN retenu à la fin de la partie 6 |
| `modeles/pipeline.json` | réglages du recadrage calibré en partie 8 (relus par la démo) |
| `modeles/yolov8n-face.pt` | détecteur de visages YOLOv8n (WIDER FACE) |
| `modeles/face_detection_yunet_2023mar.onnx` | détecteur de secours (YuNet, OpenCV) |
| `modeles/*.keras`, `*_historique.json` | les autres modèles entraînés ; le notebook les recharge au lieu de les réentraîner |

## Lancer la démo sur l'ordinateur de la présentation

**À faire la veille, avec internet** : l'installation prend environ 1,2 Go sur le disque. Sans carte graphique, la démo tourne autour de 15 images par seconde.

### Windows

1. Installer Python 3.11 ou 3.12 depuis https://www.python.org/downloads/ en cochant **« Add python.exe to PATH »**. **Python 3.10 ne marche pas** (la version de Keras qui a enregistré le modèle demande 3.11 minimum) ; si le 3.10 est déjà installé, on le garde et on installe le 3.12 à côté : `installer_demo.bat` choisit tout seul la bonne version.
2. Double-cliquer sur `installer_demo.bat` (5 à 10 minutes).
3. Double-cliquer sur `lancer_demo.bat`. Windows peut demander l'accès à la caméra : accepter.

### macOS ou Linux

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_demo.txt     # Linux : ajouter --index-url https://download.pytorch.org/whl/cpu pour torch
python demo_webcam.py
```

Sur macOS, autoriser le Terminal à utiliser la caméra (Réglages > Confidentialité et sécurité > Caméra).

### Touches

| Touche | Effet |
|---|---|
| `q` ou `Échap` | quitter |
| `espace` | figer l'image (pratique pour commenter les barres) |
| `b` | afficher ou cacher le panneau des probabilités |
| `m` | miroir |
| `f` | plein écran |
| `+` / `-` | seuil de détection des visages |
| `s` | capture d'écran dans `captures/` |

### Si quelque chose ne marche pas le jour J

| Problème | Solution |
|---|---|
| La caméra ne s'ouvre pas | `lancer_demo.bat --camera 1` (autre caméra), fermer Teams/Zoom qui peuvent la bloquer |
| YOLO ne se charge pas | `lancer_demo.bat --detecteur yunet` (détecteur de secours d'OpenCV) |
| Trop lent | `lancer_demo.bat --taille 320` |
| Pas de webcam du tout | `lancer_demo.bat --video video_demo.mp4`, ou une photo : `--image photo.jpg` |
| Rien ne s'installe | notebook principal sur Colab, partie 8 : la cellule `PRENDRE_PHOTO = True` prend une photo avec la webcam du navigateur |

Conseils pour la démo : bonne lumière de face, visage à 50 cm - 1 m, expressions franches. Montrer aussi un cas qui échoue (main devant la bouche : la joie disparaît), c'est exactement ce qu'a mesuré le red team.

## Réexécuter les notebooks

**Colab.** Ouvrir le notebook, Exécution > Modifier le type d'exécution > GPU T4, puis Exécution > Tout exécuter. Le notebook principal télécharge les données et entraîne tout (environ 1 h 30). Pour que le red team attaque notre modèle final, déposer `modeles/modele_final.keras` dans le panneau Fichiers de Colab, sous le nom `/content/modele_final.keras`.

**PC.** Le notebook principal tourne aussi en local avec le backend PyTorch de Keras (`KERAS_BACKEND=torch`). S'il trouve les modèles dans `modeles/`, il les recharge au lieu de les réentraîner.

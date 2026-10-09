"""Demonstration en direct du projet : webcam -> YOLO (visages) -> recadrage calibre -> CNN -> expression.

Meme pipeline que les parties 8 et 9 du notebook principal. Les reglages (facteur de recadrage, decalage vertical)
sont relus dans modeles/pipeline.json, ecrit par le notebook.

Utilisation :
    python demo_webcam.py                    webcam par defaut
    python demo_webcam.py --camera 1         autre webcam
    python demo_webcam.py --video clip.mp4   fichier video
    python demo_webcam.py --image photo.jpg  une photo

Touches : q ou Echap = quitter | espace = pause | s = capture d'ecran | m = miroir | b = barres de probabilites
          f = plein ecran | + / - = seuil de detection
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("KERAS_BACKEND", "torch")   # YOLO utilise deja PyTorch : un seul moteur a installer

import numpy as np
import cv2
import keras

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_MODELES = os.path.join(ICI, "modeles")

# couleur de chaque expression (BGR), pour les boites et les barres
COULEURS = {"Colere": (40, 40, 220), "Degout": (40, 150, 40), "Peur": (170, 60, 170), "Joie": (0, 200, 255),
            "Tristesse": (200, 120, 30), "Surprise": (0, 140, 255), "Neutre": (170, 170, 170)}


def lire_reglages(chemin):
    reglages = {"classes": list(COULEURS), "facteur": 1.0, "decalage_y": 0.0,
                "modele": "modele_final.keras", "detecteur": "yolov8n-face.pt"}
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            reglages.update(json.load(f))
    else:
        print(f"attention : {chemin} introuvable, recadrage par defaut (facteur 1, sans decalage)")
    return reglages


class DetecteurVisages:
    """YOLOv8n entraine sur WIDER FACE ; en secours, YuNet, le petit detecteur de visages d'OpenCV."""

    def __init__(self, nom, chemin_yolo, taille):
        self.taille = taille
        self.yolo = None
        if nom == "yolo":
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(chemin_yolo)
                print("detecteur : YOLO", chemin_yolo)
            except Exception as e:   # poids absents, ultralytics non installe...
                print(f"YOLO indisponible ({e}) : on passe a YuNet (OpenCV)")
        if self.yolo is None:
            chemin = os.path.join(DOSSIER_MODELES, "face_detection_yunet_2023mar.onnx")
            self.yunet = cv2.FaceDetectorYN.create(chemin, "", (320, 320), score_threshold=0.5)
            print("detecteur : YuNet (OpenCV)", chemin)

    def __call__(self, img_bgr, conf):
        if self.yolo is not None:
            r = self.yolo.predict(img_bgr, conf=conf, iou=0.5, imgsz=self.taille, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        h, w = img_bgr.shape[:2]
        self.yunet.setInputSize((w, h))
        self.yunet.setScoreThreshold(conf)
        _, faces = self.yunet.detect(img_bgr)
        if faces is None:
            return np.zeros((0, 4), "float32"), np.zeros(0, "float32")
        x, y, bw, bh = faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3]
        return np.stack([x, y, x + bw, y + bh], 1).astype("float32"), faces[:, -1].astype("float32")


def recadrer(gris, boite, facteur, dy):
    # carre centre sur la boite (corrige du decalage vertical), de cote facteur x le plus grand cote de la boite,
    # complete en repetant les bords s'il sort de l'image, puis ramene en 48x48 (identique au notebook, partie 8.2)
    x1, y1, x2, y2 = boite
    c = facteur * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + dy * c
    n = max(int(round(c)), 1)
    a, b = int(round(cx - c / 2)), int(round(cy - c / 2))
    H, W = gris.shape
    m = max(0, -a, -b, a + n - W, b + n - H)
    if m:
        gris = cv2.copyMakeBorder(gris, m, m, m, m, cv2.BORDER_REPLICATE)
        a, b = a + m, b + m
    return cv2.resize(gris[b:b + n, a:a + n], (48, 48), interpolation=cv2.INTER_AREA if n > 48 else cv2.INTER_LINEAR)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Pipeline:
    def __init__(self, args):
        self.reglages = lire_reglages(os.path.join(DOSSIER_MODELES, "pipeline.json"))
        self.classes = self.reglages["classes"]
        self.facteur = args.facteur if args.facteur else self.reglages["facteur"]
        self.dy = self.reglages["decalage_y"] if args.facteur is None else 0.0
        chemin_modele = args.modele or os.path.join(DOSSIER_MODELES, self.reglages["modele"])
        self.modele = keras.models.load_model(chemin_modele)
        print(f"CNN : {chemin_modele} ({self.modele.count_params():,} parametres)"
              f" | recadrage : facteur {self.facteur:.2f}, decalage {self.dy:+.3f}")
        self.detecteur = DetecteurVisages(args.detecteur, os.path.join(DOSSIER_MODELES, self.reglages["detecteur"]),
                                          args.taille)
        self.conf = args.conf
        self.lissage = args.lissage
        self.pistes = []

    def predire(self, visages48):
        x = np.stack(visages48).astype("float32")[..., None] / 255   # meme pretraitement qu'a l'entrainement
        return np.asarray(keras.ops.convert_to_numpy(self.modele(x, training=False)))

    def analyser(self, img_bgr, suivre=True):
        boites, confs = self.detecteur(img_bgr, self.conf)
        if len(boites) == 0:
            self.pistes = []
            return []
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        visages = [recadrer(gris, b, self.facteur, self.dy) for b in boites]
        probas = self.predire(visages)
        res = [{"boite": b, "confiance": float(s), "probas": p, "visage": v}
               for b, s, p, v in zip(boites, confs, probas, visages)]
        if suivre:
            # suivi simple : on reprend la piste de l'image precedente qui recouvre le plus ce visage, et on lisse
            for r in res:
                proche = max(self.pistes, key=lambda p: iou(p["boite"], r["boite"]), default=None)
                if proche is not None and iou(proche["boite"], r["boite"]) > 0.3:
                    r["probas"] = self.lissage * proche["probas"] + (1 - self.lissage) * r["probas"]
            self.pistes = res
        return res


def texte(img, t, org, echelle=0.6, couleur=(255, 255, 255), epaisseur=1, fond=None):
    (tw, th), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, echelle, epaisseur)
    if fond is not None:
        cv2.rectangle(img, (org[0] - 3, org[1] - th - 5), (org[0] + tw + 3, org[1] + base), fond, -1)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur, epaisseur, cv2.LINE_AA)


def dessiner(img, res, classes, barres=True, infos=""):
    img = img.copy()
    e = max(1, img.shape[1] // 640)
    for r in res:
        x1, y1, x2, y2 = r["boite"].astype(int)
        k = int(r["probas"].argmax())
        coul = COULEURS.get(classes[k], (0, 220, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), coul, 2 * e)
        texte(img, f"{classes[k]} {100 * r['probas'][k]:.0f}%", (x1 + 3, max(y1 - 6, 20)), 0.6 * e, (0, 0, 0), e, fond=coul)
    if infos:
        texte(img, infos, (10, 25), 0.55 * e, (255, 255, 255), e, fond=(0, 0, 0))
    texte(img, "expression affichee, pas l'emotion ressentie", (10, img.shape[0] - 12), 0.45 * e, (220, 220, 220), e, fond=(0, 0, 0))
    if not barres:
        return img
    # panneau a droite : les 7 probabilites du visage principal (le plus grand) et ce que voit le CNN
    H = img.shape[0]
    L = max(260, H // 2)
    panneau = np.full((H, L, 3), 30, "uint8")
    texte(panneau, "visage principal", (12, 30), 0.6, (255, 255, 255), 1)
    if res:
        r = max(res, key=lambda r: (r["boite"][2] - r["boite"][0]) * (r["boite"][3] - r["boite"][1]))
        cote = min(L - 24, H // 3)
        vu = cv2.resize(r["visage"], (cote, cote), interpolation=cv2.INTER_NEAREST)
        panneau[45:45 + cote, 12:12 + cote] = cv2.cvtColor(vu, cv2.COLOR_GRAY2BGR)
        texte(panneau, "entree du CNN (48x48)", (12, 45 + cote + 18), 0.45, (180, 180, 180), 1)
        y0 = 45 + cote + 40
        h = max(14, min(28, (H - y0 - 10) // 7 - 6))
        for k, (c, p) in enumerate(zip(classes, r["probas"])):
            y = y0 + k * (h + 6)
            cv2.rectangle(panneau, (100, y), (100 + int((L - 160) * p), y + h), COULEURS.get(c, (200, 200, 200)), -1)
            texte(panneau, c, (12, y + h - 3), 0.5, (255, 255, 255), 1)
            texte(panneau, f"{100 * p:.0f}%", (L - 52, y + h - 3), 0.5, (255, 255, 255), 1)
    else:
        texte(panneau, "aucun visage", (12, 70), 0.6, (180, 180, 180), 1)
    return np.hstack([img, panneau])


def ouvrir_camera(index):
    api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY   # DirectShow s'ouvre plus vite sous Windows
    cam = cv2.VideoCapture(index, api)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam


def main():
    p = argparse.ArgumentParser(description="Demo : reconnaissance d'expressions faciales en direct")
    p.add_argument("--camera", type=int, default=0, help="numero de la webcam (0 par defaut)")
    p.add_argument("--video", help="analyser un fichier video au lieu de la webcam")
    p.add_argument("--image", help="analyser une photo")
    p.add_argument("--sortie", help="enregistrer le resultat (image ou video .mp4)")
    p.add_argument("--modele", help="chemin d'un autre modele .keras")
    p.add_argument("--detecteur", choices=["yolo", "yunet"], default="yolo", help="yunet : detecteur de secours d'OpenCV")
    p.add_argument("--conf", type=float, default=0.5, help="seuil de confiance du detecteur")
    p.add_argument("--taille", type=int, default=480, help="taille d'image donnee a YOLO (plus petit = plus rapide)")
    p.add_argument("--facteur", type=float, help="remplace le facteur de recadrage calibre")
    p.add_argument("--lissage", type=float, default=0.6, help="0 = pas de lissage, 0.9 = tres lisse")
    p.add_argument("--sans-fenetre", action="store_true", help="ne rien afficher (avec --sortie)")
    args = p.parse_args()

    pipe = Pipeline(args)
    classes = pipe.classes

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"impossible de lire {args.image}")
        res = pipe.analyser(img, suivre=False)
        for r in res:
            print({c: f"{100 * q:.0f}%" for c, q in zip(classes, r["probas"]) if q > 0.05})
        sortie = dessiner(img, res, classes, infos=f"{len(res)} visage(s)")
        if args.sortie:
            cv2.imwrite(args.sortie, sortie)
        if not args.sans_fenetre:
            cv2.imshow("Expressions faciales", sortie)
            cv2.waitKey(0)
        return

    source = args.video if args.video else args.camera
    flux = cv2.VideoCapture(args.video) if args.video else ouvrir_camera(args.camera)
    if not flux.isOpened():
        sys.exit(f"impossible d'ouvrir {source} (essayer --camera 1, ou verifier les autorisations de la camera)")
    miroir = not args.video
    barres, pause, plein_ecran = True, False, False
    ecrivain = None
    fenetre = "Expressions faciales (q pour quitter)"
    if not args.sans_fenetre:
        cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
    fps, t_prec, n = 0.0, time.time(), 0
    sortie = None
    while True:
        if not pause or sortie is None:
            ok, frame = flux.read()
            if not ok:
                break
            if miroir:
                frame = cv2.flip(frame, 1)
            res = pipe.analyser(frame)
            t = time.time()
            fps = 0.9 * fps + 0.1 / max(t - t_prec, 1e-6) if n else 1 / max(t - t_prec, 1e-6)
            t_prec, n = t, n + 1
            sortie = dessiner(frame, res, classes, barres,
                              f"{fps:4.1f} img/s | {len(res)} visage(s) | seuil {pipe.conf:.2f}")
            if args.sortie:
                if ecrivain is None:
                    h, w = sortie.shape[:2]
                    ecrivain = cv2.VideoWriter(args.sortie, cv2.VideoWriter_fourcc(*"mp4v"),
                                               flux.get(cv2.CAP_PROP_FPS) or 15, (w, h))
                ecrivain.write(sortie)
        if args.sans_fenetre:
            continue
        cv2.imshow(fenetre, sortie)
        touche = cv2.waitKey(1) & 0xFF
        if touche in (ord("q"), 27):
            break
        elif touche == ord(" "):
            pause = not pause
        elif touche == ord("m"):
            miroir = not miroir
        elif touche == ord("b"):
            barres = not barres
        elif touche == ord("s"):
            os.makedirs(os.path.join(ICI, "captures"), exist_ok=True)
            nom = os.path.join(ICI, "captures", time.strftime("capture_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(nom, sortie)
            print("capture :", nom)
        elif touche == ord("f"):
            plein_ecran = not plein_ecran
            cv2.setWindowProperty(fenetre, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if plein_ecran else cv2.WINDOW_NORMAL)
        elif touche in (ord("+"), ord("=")):
            pipe.conf = min(0.95, pipe.conf + 0.05)
        elif touche == ord("-"):
            pipe.conf = max(0.05, pipe.conf - 0.05)
    flux.release()
    if ecrivain is not None:
        ecrivain.release()
        print("video enregistree :", args.sortie)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

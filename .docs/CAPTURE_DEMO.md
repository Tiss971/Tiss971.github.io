# Capture live → segmentation → reconstruction 3D (100% navigateur)

Démo interactive : [`capture-demo.html`](../capture-demo.html). Le visiteur active sa webcam, tourne sur
lui-même pendant ~12 secondes, et voit un nuage de points 3D se construire en direct — aucun upload,
aucun backend, tout tourne dans le navigateur.

## Pourquoi pas un "vrai" 3D Gaussian Splatting entraîné ?

Un vrai 3DGS s'obtient par descente de gradient (optimisation itérative des paramètres des gaussiennes)
à partir de poses caméra précises, typiquement calculées par COLMAP à partir de dizaines/centaines de
photos d'une scène **statique** vue sous des angles **différents**.

Ici la situation est inversée : la caméra est fixe, c'est le sujet qui tourne. Deux limites en découlent :

1. **Pas d'outil de training différentiable mature 100% navigateur.** Recherché avant d'implémenter :
   le sujet est actif en 2026 (ex. papier [WebSplatter](https://arxiv.org/pdf/2602.03207), arXiv
   2602.03207) mais rien qui s'installe et fonctionne de façon fiable en quelques jours.
2. **[Brush](https://github.com/ArthurBrussee/brush)** (Rust/Burn/WebGPU) supporte bien le training en
   navigateur, mais attend des poses caméra qualité COLMAP/Nerfstudio, n'expose aucune API JS/WASM
   documentée pour l'embedding dans une page custom, et son support WebGPU est limité à Chrome 134+.
   Piste retenue pour une **v2** éventuelle (export des frames + poses vers un pipeline offline), pas
   dans le scope de cette démo live.

**Décision assumée** : construire une reconstruction **heuristique**, pas un vrai training. C'est un
choix d'ingénierie explicite — savoir doser l'ambition technique face au temps disponible plutôt que
prétendre à un résultat qu'on ne peut pas livrer proprement.

## Comment ça marche

1. **Webcam** — `getUserMedia`, flux affiché en direct dans la page
2. **Segmentation temps réel** — [`ImageSegmenter`](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
   de MediaPipe Tasks Vision (modèle `selfie_segmenter`, tourne en WASM dans le navigateur) : masque de
   confiance par pixel, seuil à 0.5 pour isoler le sujet du fond
3. **Détection de présence** — [`PoseLandmarker`](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
   (modèle `pose_landmarker_lite`) sert de filtre qualité : une frame n'est retenue que si une personne
   y est détectée. **Il n'est pas utilisé pour calculer l'angle de rotation** — une caméra monoculaire
   fixe ne peut pas récupérer de façon fiable un angle de rotation complet à partir de la seule
   orientation des épaules (ambiguïté face/dos, pas d'information de profondeur absolue). C'est une
   limite géométrique de fond, pas un raccourci d'implémentation.
4. **Angle de rotation** — estimé par interpolation linéaire sur la durée de capture (hypothèse
   "plateau tournant à vitesse constante" — le sujet tourne de 360° sur ~12s)
5. **Placement heuristique des points** — pour chaque frame échantillonnée (~350ms d'intervalle), les
   pixels foreground sont posés comme un « billboard » plat **centré sur l'origine** et tourné autour de
   l'axe Y de l'angle estimé de la frame ; couleur = couleur du pixel vidéo. Toutes les frames se
   superposent au centre → un éventail façon plateau tournant qui se lit comme une figure unique centrée
   (et non une ronde de copies). Aucune optimisation, aucune descente de gradient — un placement
   géométrique direct (voir [`buildPointCloud`](../capture-demo.js))
6. **Rendu** — `THREE.Points` avec un sprite circulaire dégradé (généré par canvas) pour un rendu
   "gaussien-like" léger, contrôles orbit (glisser + molette)

## Limites connues
- Reconstruction approximative : bruit visible, pas de vraie géométrie 3D cohérente sous tous les angles
- Sensible à la vitesse de rotation réelle vs. l'hypothèse de vitesse constante
- Qualité de segmentation dépendante de l'éclairage et du contraste sujet/fond
- Pas de désocclusion : les zones jamais vues par la webcam restent vides

## Pistes d'évolution (hors scope actuel)
- Export des frames + masques + poses estimées vers un vrai pipeline 3DGS offline (COLMAP + training GPU)
- Expérimenter Brush une fois son support d'embedding JS documenté, ou avec des poses de meilleure qualité
- Fusion multi-passes (plusieurs tours de webcam) pour densifier le nuage de points

## Fichiers
- [`capture-demo.html`](../capture-demo.html) — UI (webcam, progression, overlay stats, résultat)
- [`capture-demo.js`](../capture-demo.js) — pipeline complet : capture, segmentation, reconstruction, rendu

## Test automatisé

Un harnais [`test/run_capture_test.py`](../test/run_capture_test.py) pilote le Chrome installé en headless
avec une **fausse webcam** (`--use-file-for-fake-video-capture`), permettant de valider le pipeline sans
caméra physique. Dépendances : `pip install playwright imageio-ffmpeg` (réutilise Chrome via
`channel="chrome"`, pas de téléchargement de navigateur).

```
python test/run_capture_test.py                 # synthétique (valide l'infra ; 0 frame capturée)
python test/run_capture_test.py --clip test/turn.mp4   # test complet avec un vrai clip 360°
```

Le test complet exige un clip d'une **vraie personne** tournant sur 360° (`test/turn.mp4`) : le
`PoseLandmarker` ne retient une frame que si une personne est détectée. Screenshots sauvegardés dans
`test/out/`.

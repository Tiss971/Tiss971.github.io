# Capture live → segmentation → reconstruction 3D (100% navigateur)

Démo interactive : [`capture-demo.html`](../capture-demo.html). Le visiteur active sa webcam, bouge
librement devant (il peut montrer chaque profil, sans tour complet) pendant quelques secondes, et voit un
nuage de points 3D se construire en direct — aucun upload, aucun backend, tout tourne dans le navigateur.

## Pourquoi pas un "vrai" 3D Gaussian Splatting entraîné ?

Un vrai 3DGS s'obtient par descente de gradient (optimisation itérative des paramètres des gaussiennes)
à partir de poses caméra précises, typiquement calculées par COLMAP à partir de dizaines/centaines de
photos d'une scène **statique** vue sous des angles **différents**.

Ici la situation est inversée : la caméra est fixe, c'est le sujet qui bouge librement devant. Deux limites
en découlent :

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
   y est détectée.
4. **Angle par frame — head-pose + flux optique** — aucune hypothèse de plateau tournant ni de 360° :
   l'utilisateur bouge librement (il peut montrer une joue puis l'autre). L'angle de chaque frame est
   **mesuré**, pas supposé :
   - **Head-pose** : [`FaceLandmarker`](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
     fournit une matrice de transformation faciale d'où on extrait le **yaw** (rotation de la tête) — un
     angle réel, absolu, par frame. Fonctionne tant que le visage est visible (~±70° de yaw).
   - **Repli flux optique (calibré)** : quand aucun visage n'est détecté (profil marqué, occlusion), on
     **intègre le déplacement horizontal signé** du sujet segmenté entre frames (corrélation croisée 1-D
     des profils de colonnes). L'échelle pixels→radians est **calibrée sur les ancres head-pose** — entre
     deux frames à visage, le vrai Δyaw est connu, donc on ajuste le facteur pour que le flux colle. Ça
     corrige le **signe et l'amplitude** au lieu d'une constante devinée (qui sous-tournait ~2,5×).

   Voir [`estimateFrameAngles`](../capture-demo.js). Une caméra monoculaire fixe ne peut pas récupérer un
   angle métrique parfait — c'est une limite géométrique de fond, pas un raccourci d'implémentation.
5. **Reconstruction — visual hull (shape-from-silhouette)** — une grille de voxels (64×96×64) centrée sur
   l'origine est **sculptée** : chaque voxel est tourné dans l'espace caméra de chaque frame (selon son
   angle estimé) puis projeté orthographiquement sur le masque ; on ne le garde que s'il tombe dans la
   silhouette dans ≥70% des vues où il se projette (**intersection des cônes de silhouettes**). Les vues
   de profil (yaw) contraignent la profondeur, les vues de face la largeur/hauteur ; couleur = vue la plus
   frontale. Nettoyage : (a) par frame, seul le **plus grand composant connexe** du masque est conservé
   (anti-parasites) ; (b) les voxels ayant trop peu de voisins occupés sont jetés (anti-specks). Voir
   [`buildPointCloud`](../capture-demo.js). Aucune descente de gradient.
6. **Rendu — vraies gaussiennes (SparkJS)** — chaque voxel survivant devient une **gaussienne isotrope**,
   rendue par [SparkJS](https://sparkjs.dev) (`SplatMesh` + `SparkRenderer`, la même stack `.sog` que le
   fond du portfolio) : du vrai 3D gaussian splatting, pas une imitation. Repli automatique sur
   `THREE.Points` si SparkJS ne s'initialise pas. Contrôles orbit (glisser + molette).

## Export COLMAP (vers un pipeline 3DGS offline)

Le bouton **« Exporter (COLMAP) »** génère, 100% dans le navigateur, un **ZIP** prêt pour un entraîneur
3DGS (gsplat, Brush, 3DGS Inria) :
- `images/frame_XXX.png` — les frames capturées en **pleine résolution** (640×480) ;
- `cameras.txt` — une caméra `PINHOLE` (focale déduite d'un FOV supposé de 60°) ;
- `images.txt` — **une pose caméra par frame** (quaternion + translation, convention world→camera COLMAP),
  obtenue en convertissant le plateau tournant (caméra fixe / sujet qui tourne) en « caméra qui orbite un
  objet fixe » ;
- `points3D.txt` vide (init aléatoire ou depuis le hull).

Writer ZIP maison (méthode *store* + CRC32), aucune dépendance — cohérent avec le « tout navigateur ».
**Réserve** : le format est garanti bien-formé (validé : quaternions unitaires, comptes cohérents, CRC OK),
mais la **convention géométrique** (sens d'orbite, flip d'axes) n'est vérifiable que dans un vrai run
gsplat/COLMAP — deux constantes (`COLMAP_AZIMUTH_SIGN`, `FACE_YAW_SIGN`) corrigent un miroir en une ligne.
Les poses monoculaires restent approximatives (yaw seul, extrapolation quand le visage est caché).

## Limites connues
- **Qualité du hull = qualité des poses** : l'intersection de cônes de silhouettes est très sensible à la
  précision des angles et à l'étalement angulaire. Des angles faux (flux qui dérive, couverture partielle)
  **sur-sculptent** → forme dégénérée. Le cas propre = visage bien visible + profils marqués (head-pose fiable).
- **Head-pose limité au visage visible** (~±70° de yaw) ; au-delà, on dépend du flux optique
- **Flux optique** : échelle pixels→radians heuristique + intégration → dérive possible sur de longues
  séquences sans visage
- **Projection orthographique** : approximation (la webcam est en perspective)
- Qualité de segmentation dépendante de l'éclairage et du contraste sujet/fond
- Pas de désocclusion : les zones jamais vues restent creuses

## Pistes d'évolution (hors scope actuel)
- **Poses plus robustes** : meilleure calibration/lissage du flux, ou head-pose multi-frames, pour un hull
  stable même en couverture partielle
- **Gaussiennes anisotropes** alignées sur la surface locale (au lieu d'isotropes) pour un rendu plus fin
- **Entraîner un vrai 3DGS offline** à partir de l'export COLMAP (gsplat/Brush sur GPU), puis réimport du
  `.sog` entraîné dans le viewer SparkJS — le « vrai » 3DGS, la démo servant de front-end de capture
- Fusion multi-passes pour densifier le volume

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
python test/run_capture_test.py --clip test/turn.mp4   # test complet avec un vrai clip
```

Le test complet exige un clip d'une **vraie personne** (`test/turn.mp4`) : le `PoseLandmarker` ne retient
une frame que si une personne est détectée. Screenshots sauvegardés dans `test/out/`.

> Trois modèles MediaPipe se chargent (segmentation, pose, visage). Le `FaceLandmarker` ne mesure un yaw
> que lorsque le visage fait face à la caméra ; sinon la frame bascule sur le flux optique. La sonde
> `window.__angleStats` (imprimée par le harnais) indique la répartition face/flux — p. ex. `face=7, flow=9`.

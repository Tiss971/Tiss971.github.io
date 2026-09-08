# Capture live → segmentation → reconstruction 3D (100% navigateur)

Démo interactive : [`capture-demo.html`](../demo/capture/capture-demo.html). Le visiteur active sa webcam, bouge
librement devant (il peut montrer chaque profil, sans tour complet) pendant quelques secondes, et voit un
nuage de points 3D se construire en direct — aucun upload, aucun backend, tout tourne dans le navigateur.

![Résultat d'une capture — visual hull rendu en gaussiennes anisotropes SparkJS](../assets/capture-demo-result.png)

**État** : pipeline live (webcam → segmentation → visual hull → rendu gaussien) fonctionnel et testé,
recadré sur **tête+cou** (le torse, mal contraint géométriquement, est volontairement exclu — voir
[TODO](#todo)) ; **trois modes de reconstruction supplémentaires** basés sur MediaPipe `FaceLandmarker`
sélectionnables en direct via un bouton (cycle hull → mesh → dense → **fused** → hull) — mesh (478 points,
rayon de gaussienne adaptatif par densité locale), dense (grille fine sur toute la région du visage, profondeur
d'un **vrai modèle monoculaire** — Depth Anything v2 — plutôt qu'interpolée, le résultat le plus proche d'un
vrai visage à ce jour) et **fused** (dense + fusion multi-vues, même modèle de profondeur monoculaire par
vue, calibré indépendamment — le seul des trois qui exploite vraiment la rotation de l'utilisateur) —
implémentés et validés ; export COLMAP fonctionnel mais convention
géométrique non vérifiée sur un vrai run gsplat ; training 3DGS réel hors-scope (voir ci-dessous), non
commencé.

![Mode dense — grille fine avec profondeur d'un vrai modèle monoculaire (Depth Anything v2)](../assets/capture-demo-dense-result.png)

- [Pourquoi pas un training 3DGS réel ?](#pourquoi-pas-un-vrai-3d-gaussian-splatting-entraîné-)
- [Comment ça marche](#comment-ça-marche)
- [Export COLMAP](#export-colmap-vers-un-pipeline-3dgs-offline)
- [Limites connues](#limites-connues)
- [TODO](#todo)
- [Pistes d'évolution](#pistes-dévolution-hors-scope-actuel)
- [Fichiers](#fichiers)
- [Test automatisé](#test-automatisé)

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
4. **Angle par frame — head-pose ancré, flux interpolé, torse en renfort** — aucune hypothèse de plateau
   tournant ni de 360° : l'utilisateur bouge librement (il peut montrer une joue puis l'autre). L'angle de
   chaque frame est **mesuré**, pas supposé. Trois signaux, avec une hiérarchie de confiance dictée par une
   vérification empirique (voir plus bas) :
   - **Head-pose = étalon** : [`FaceLandmarker`](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
     donne le **yaw** de la tête (matrice de transformation faciale) — mesure bien posée, sans ambiguïté
     avant/arrière, tant que le visage est visible (~±70°). C'est l'**ancre absolue** de référence.
   - **Flux optique = colonne vertébrale lisse** : le déplacement horizontal signé du sujet segmenté entre
     frames (corrélation croisée 1-D des profils de colonnes) donne un signal **continu et sans ambiguïté**
     (mais d'échelle inconnue). Entre deux ancres visage, l'angle est **interpolé proportionnellement au flux
     accumulé** ; l'échelle pixels→radians est calibrée sur les paires de visages (médiane robuste).
   - **Pose du torse = renfort** : [`PoseLandmarker`](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
     donne le yaw des épaules (`atan2(dz, dx)` sur les `worldLandmarks`). Il précise l'angle quand le visage
     est absent — **mais seulement s'il concorde** avec la base flux/visage : la profondeur `z` des épaules
     de MediaPipe **bascule avant/arrière près du profil** (sauts de 150–300°), donc ces branches miroir sont
     **rejetées**. Voir [`yawFromPoseResult`](../demo/capture/capture-demo.js).

   Voir [`estimateFrameAngles`](../demo/capture/capture-demo.js). Limite géométrique de fond (pas un raccourci) :
   voir [Limites connues](#limites-connues) pour l'ambiguïté avant/arrière quand le sujet tourne le dos.

   **Capture guidée par couverture** — plutôt qu'un minuteur fixe, la capture s'arrête quand assez de
   secteurs d'azimut distincts ont été vus (un meilleur étalement angulaire = un meilleur hull). L'arc de
   progression se remplit selon la **couverture atteinte** (pas le temps), un coach invite à continuer de
   tourner, et un plafond de temps garantit la terminaison. Voir [`startCapture`](../demo/capture/capture-demo.js).
5. **Reconstruction — visual hull (shape-from-silhouette)** — une grille de voxels (64×96×64) centrée sur
   l'origine est **sculptée** : chaque voxel est tourné dans l'espace caméra de chaque frame (selon son
   angle estimé) puis projeté orthographiquement sur le masque ; on ne le garde que s'il tombe dans la
   silhouette dans ≥70% des vues où il se projette (**intersection des cônes de silhouettes**). Les vues
   de profil (yaw) contraignent la profondeur, les vues de face la largeur/hauteur ; couleur = vue la plus
   frontale. Nettoyage : (a) par frame, seul le **plus grand composant connexe** du masque est conservé
   (anti-parasites) ; (b) les voxels ayant trop peu de voisins occupés sont jetés (anti-specks) ; (c) la
   sculpture est **recadrée sur tête+cou** (seuil dérivé de la ligne d'épaules mesurée à chaque capture,
   `carveMinY`/`MIN_CARVE_Y`) — le torse n'est pas contraint par de vraies vues de profil (seul le visage
   ancre l'angle précisément) et dégénère en forme conique ; la démo n'ayant pas
   besoin du torse, cette zone est exclue plutôt que corrigée (voir [TODO](#todo)). Voir
   [`buildPointCloud`](../demo/capture/capture-demo.js). Aucune descente de gradient.
6. **Rendu — vraies gaussiennes (SparkJS)** — chaque voxel (ou sommet, en mode mesh) survivant devient une
   gaussienne, rendue par [SparkJS](https://sparkjs.dev) (`SplatMesh` + `SparkRenderer`, la même stack
   `.sog` que le fond du portfolio) : du vrai 3D gaussian splatting, pas une imitation. Repli automatique
   sur `THREE.Points` si SparkJS ne s'initialise pas. Contrôles orbit (glisser + molette).
   **Hull : gaussiennes anisotropes.** À la densité de points de ce hull (~4-5k pour tout le buste), des
   sphères isotropes lisent comme un amas de points plutôt qu'un visage. Chaque voxel gardé avec une
   normale de surface exploitable (gradient d'occupation de ses 6 voisins directs — `estimateVoxelNormal`,
   testé isolément avant intégration : mur plat, voxel intérieur, coin de cube, speck isolé) devient un
   disque aligné sur cette normale au lieu d'une sphère (`THREE.Quaternion.setFromUnitVectors`) ; un voxel
   sans normale exploitable (gradient nul — entièrement entouré) reste isotrope. **Premier réglage bien
   trop agressif** (rayon ×0.35 sur la normale, ×1.25 en tangent) : à cette densité de points, des disques
   aussi plats laissent voir le fond dès que leur face n'est pas tournée vers la caméra — résultat
   troué/filiforme, pire que les sphères. **Resserré à ×0.6/×1.0** (`ANISO_NORMAL_SCALE_MULT`/
   `ANISO_TANGENT_SCALE_MULT`) : relief du visage (nez, joues) nettement plus lisible de face **et** de
   profil, sans trou visible — voir capture ci-dessus.
   **mesh/dense/fused : même principe, normale estimée différemment.** Pas de grille de voxels dans ces
   modes (nuage de points épars ou grille de pixels), donc la normale de chaque point vient d'un **fit de
   plan local (PCA)** sur ses voisins dans un rayon donné (`MESH_NORMAL_RADIUS`/`DENSE_NORMAL_RADIUS`,
   `estimatePointCloudNormals`) — la normale est le vecteur propre de plus petite valeur propre de la
   matrice de covariance du voisinage. Recherche de voisinage par **hash spatial uniforme** (buckets de
   `radius/1.5`, comme la grille de fusion du mode fused) plutôt que force brute : le mode dense peut
   produire plusieurs milliers de points, où une recherche O(n²) type `nearestNeighborDistances`
   deviendrait trop lente. Résolution du plus petit vecteur propre en **forme fermée** (Smith 1961) plutôt
   que par itération de puissance : un cas de test (plan incliné à 45°) a révélé qu'un vecteur de départ
   peut tomber exactement orthogonal au vecteur propre visé sous une distribution de points symétrique — la
   symétrie bilatérale d'un visage rend ce cas plausible en pratique, pas seulement théorique. Un point avec
   trop peu de voisins dans le rayon reste isotrope (même repli que le hull). Réutilise directement
   `ANISO_NORMAL_SCALE_MULT`/`ANISO_TANGENT_SCALE_MULT` — validé visuellement de face et de profil sur les
   3 modes sans streaking ni trous (contrairement au premier réglage du hull, la densité de points plus
   élevée de ces modes rend le rendu robuste dès le premier essai).
7. **Modes alternatifs — mesh canonique MediaPipe** — bouton qui cycle **hull → mesh → dense → fused →
   hull** (désactivé si aucun visage n'a été détecté pendant la capture) : re-rendu instantané des **mêmes
   frames déjà capturées**, sans nouvelle capture. Pas de torse dans ces trois modes (cohérent avec le
   recadrage déjà fait sur le hull).
   - **mesh** — géométrie = les 478 landmarks d'une **seule frame de référence** (la plus frontale,
     `|yaw|` minimal) — topologie fixe, donc **jamais informe** contrairement au hull (au pire, aussi
     imparfait que cette unique détection, mais toujours en forme de visage). Couleur = **fusion
     multi-vues exacte par sommet** : pour chacun des 478 points, la frame où il est vu le plus de face
     (`z` le plus petit = le plus proche caméra) fournit sa couleur, échantillonnée directement au pixel
     du landmark dans cette frame. Rayon de gaussienne **adaptatif** par densité locale de landmarks (petit
     autour des yeux, plus grand sur les joues). Voir [`buildFaceMesh`](../demo/capture/capture-demo.js).
   - **dense** — même frame de référence, mais au lieu de 478 points, une **grille fine de pixels** sur
     toute la région du visage (limitée par la bbox des landmarks + le masque de segmentation) ;
     profondeur d'un **vrai modèle de depth monoculaire** (Depth Anything v2, calibré sur les landmarks —
     voir section dédiée plus bas), avec repli sur l'interpolation entre landmarks (IDW) si le modèle est
     indisponible. Couleur échantillonnée directement dans l'image de la frame de référence. Résultat
     nettement plus proche d'un vrai visage (peau, yeux, sourcils continus, vrai relief) que le mesh épars
     — voir capture ci-dessus. **Mono-frame comme le mesh : n'exploite pas la rotation** — géométrie ET
     couleur viennent d'une seule photo. Voir [`buildDenseFaceMesh`](../demo/capture/capture-demo.js).
   - **fused** — le mode dense qui exploite vraiment la rotation : fusionne la frame de référence avec
     jusqu'à 3 vues supplémentaires réparties sur l'éventail d'angles capturé, chacune avec sa **propre**
     inférence de profondeur calibrée indépendamment, alignée dans l'espace local de la référence par sa
     **matrice de rotation complète** (`facialTransformationMatrix`) — pas besoin de Kabsch/SVD, une
     matrice de rotation est orthogonale donc son inverse est juste sa transposée. Comble ce que la seule
     vue de face ne peut pas voir (côtés du nez, joues) ; les points de vues différentes tombant au même
     endroit sont **moyennés** (position + couleur) plutôt que laissés en doublons. Résultat visiblement
     plus large/complet que le mode dense seul, sans halo ni doublon — voir capture ci-dessous.
     **Densité de grille alignée sur la référence** (`DENSE_FUSION_STEP_MULT=1`) : les vues secondaires
     échantillonnaient à l'origine une grille 2× plus grossière que la référence (`DENSE_FUSION_STEP_MULT=2`,
     pensé pour limiter le coût — les vues secondaires ne couvrent qu'une bande extérieure), mais la
     texture visiblement plus grossière de cette bande, collée à la grille fine de la référence, créait une
     couture nette à leur jonction — lue comme du bruit/"brouillon" en zoomant, alors que le mode dense seul
     (100% référence) n'a jamais cette coupure. Confirmé par comparaison zoomée avant/après avec la même
     capture. Voir [`buildDenseFusedFaceMesh`](../demo/capture/capture-demo.js).

![Mode fused — dense + fusion multi-vues, chaque vue calibrée indépendamment sur le vrai modèle de profondeur](../assets/capture-demo-fused-result.png)

## Export COLMAP (vers un pipeline 3DGS offline)

Le bouton **« Exporter (COLMAP) »** génère, 100% dans le navigateur, un **ZIP** prêt pour un entraîneur
3DGS (gsplat, Brush, 3DGS Inria) :
- `images/frame_XXX.png` — les frames capturées en **pleine résolution** (640×480) ;
- `cameras.txt` — une caméra `PINHOLE` (focale déduite d'un FOV supposé de 60°) ;
- `images.txt` — **une pose caméra par frame** (quaternion + translation, convention world→camera COLMAP),
  obtenue en convertissant le plateau tournant (caméra fixe / sujet qui tourne) en « caméra qui orbite un
  objet fixe » ;
- `points3D.txt` — le nuage du **mode actuellement affiché** (position + couleur RGB), qui sert de
  graine au 3DGS au lieu d'une init aléatoire ; mêmes coordonnées monde que les poses (objet centré
  sur l'origine). Suit le mode à l'écran depuis que `lastCloud` est renseigné dans
  `initSceneAndRender` (commun aux 4 modes) plutôt que seulement dans `buildPointCloud` (hull) —
  avant ce changement, l'export restait toujours le hull même en visualisant dense/fused/mesh.
  Vérifié : export en mode dense → `points3D.txt` a bien le nombre de points de dense, pas du hull.

Writer ZIP maison (méthode *store* + CRC32), aucune dépendance — cohérent avec le « tout navigateur ».
**Réserve** : le format est garanti bien-formé (validé : quaternions unitaires, comptes cohérents, CRC OK),
mais la **convention géométrique** (sens d'orbite, flip d'axes) n'est vérifiable que dans un vrai run
gsplat/COLMAP — voir [Limites connues](#limites-connues) pour les constantes de signe qui corrigent un
miroir éventuel.

### Export/import interne (JSON) — round-trip dans le viewer lui-même

Contrairement à l'export COLMAP (usage externe, one-way vers un entraîneur 3DGS), **« Sauver
(JSON) »**/**« Charger (JSON) »** permettent de sauvegarder et recharger le nuage rendu **dans ce
viewer**, sans capture ni webcam. Utile pour comparer des runs (ex. avec
`test/reconstruction_quality.py`) sans tout refaire à chaque fois.

Le fichier JSON contient exactement les entrées de [`initSceneAndRender`](../demo/capture/capture-demo.js)
(`mode, positions, colors, splatScale, perPointScale, perPointNormal`) — recharger appelle cette
même fonction avec les données sauvegardées, donc les gaussiennes anisotropes (scale/quaternion
par splat) sont reconstruites par le même code que l'affichage normal, pas dupliquées dans un
chemin d'import séparé. Contrairement à l'export COLMAP (toujours limité au hull avant le fix
ci-dessus), ceci capture n'importe quel mode.

Le bouton « Charger (JSON) » est en position fixe (coin haut-droit), accessible à **toute étape**
(avant même d'activer la webcam) — contrairement à « Sauver » qui vit dans le panneau de résultat
(n'a de sens qu'une fois quelque chose de rendu). Charger un fichier désactive le bouton de cycle
de mode (le nuage chargé n'a plus les données sources pour re-render un autre mode) et bascule
l'UI directement vers l'état résultat, sans passer par la capture.

**Piège rencontré** : `reconMode` (variable module) doit être mis à jour vers `pts.mode` **avant**
d'appeler `initSceneAndRender`, sinon le nuage rechargé est étiqueté avec l'ancienne valeur de
`reconMode` (vérifié : sans ce correctif, un export "fused" rechargeait avec `mode: "hull"`, la
valeur par défaut). Autre piège : le style `class="hidden"` sur l'input file ne fait rien dans ce
fichier — `.hidden` n'existe qu'en règles scopées par ID (`#demo-panel.hidden`, etc.), pas en
classe générique ; l'input caché utilise `style="display:none"` à la place.

## Limites connues
- **Qualité du hull = qualité des poses + couverture** : l'intersection de cônes de silhouettes est très
  sensible à la précision des angles et à l'étalement angulaire. Des angles faux ou une bande d'azimut trop
  étroite **sur-sculptent** ou laissent une **dalle bouffie** (profondeur non contrainte) — c'est ce qui
  dégénérait le torse en forme conique (voir [TODO](#todo)) ; la sculpture est maintenant recadrée sur
  tête+cou (`MIN_CARVE_Y`) où la couverture angulaire du visage suffit à bien contraindre la profondeur. La
  capture guidée par couverture atténue le problème sur la zone conservée.
- **Ambiguïté avant/arrière (fond monoculaire)** : quand le sujet tourne le dos, le visage disparaît et la
  profondeur `z` des épaules **bascule de branche** (sauts de 150–300°, vérifié). On rejette ces branches
  miroir et on interpole le flux entre ancres visage ; le résultat reste **lisse** mais la portion sans
  visage est approximative. Le régime propre = visage visible tout du long (rotation partielle « une joue
  puis l'autre »), là où head-pose et torse concordent.
- **Head-pose limité au visage visible** (~±70° de yaw) ; c'est l'ancre de référence.
- **Pose du torse** : renfort seulement quand elle concorde avec la base flux/visage.
- **Flux optique** : lisse mais d'échelle imparfaite (la rotation n'est pas une simple translation
  horizontale) → l'interpolation entre ancres peut sous/sur-tourner sur les longs segments sans visage.
- **Projection orthographique** : approximation (la webcam est en perspective)
- Qualité de segmentation dépendante de l'éclairage et du contraste sujet/fond
- Pas de désocclusion : les zones jamais vues restent creuses
- **Constantes de signe non validées en conditions réelles** : `BODY_YAW_SIGN` (torse, capture live),
  `FACE_YAW_SIGN` (visage), `COLMAP_AZIMUTH_SIGN` (export) corrigent chacune un miroir potentiel en une
  ligne, mais leur signe correct dépend d'un vrai run navigateur / gsplat pour être confirmé. Voir
  [capture-demo.js](../demo/capture/capture-demo.js).
- **Modes mesh/dense/fused** : visage seul (pas de torse/cheveux complets). **mesh** et **dense**
  n'exploitent pas la rotation de l'utilisateur (géométrie et couleur d'une seule frame de référence —
  équivalent à une photo figée) ; **fused** l'exploite en fusionnant jusqu'à 4 vues, alignées par leur
  matrice de rotation complète (tangage/roulis inclus, pas seulement le lacet). Reste limité à 4 vues max.
  Dense et fused utilisent tous les deux un **vrai modèle de profondeur monoculaire** (Depth Anything v2,
  voir section dédiée ci-dessous) quand il est disponible, avec repli sur l'interpolation entre landmarks
  (IDW) sinon (pas de réseau, pas de WASM/WebGPU). En mode mesh, 478 points restent visiblement discrets
  malgré le rayon de gaussienne adaptatif (pointillisme dans les zones les moins denses, ex. joues) —
  atténué en dense/fused par la grille de pixels fine.
- **Les gaussiennes SparkJS n'ont pas de face cachée (pas de backface culling)** — une capture statique
  de face ne peut donc pas détecter un problème de profondeur/désalignement : la couleur seule suffit à
  "ressembler à un visage" même si la géométrie est fausse en Z. Toute vérification touchant à la
  profondeur (signes, fusion multi-vues, alignement) doit être faite **en tournant la caméra** (voir
  `side_view_check.py`, scratchpad de session, réutilisable), pas seulement sur le screenshot par défaut.

## TODO
Amélioration de la qualité des poses/du hull, par priorité (diagnostic confirmé sur une capture réelle,
voir [Limites connues](#limites-connues)) :

1. ~~**Resserrer/désactiver le renfort torse**~~ — **Fait** ([capture-demo.js:52-53](../demo/capture/capture-demo.js#L52-L53)) :
   `BODY_YAW_REINFORCE = false` (renfort désactivé, activable via un flag) + tolérance d'accord séparée
   `BODY_AGREE_TOL` (20°, au lieu de partager `JUMP_TOL`=70° avec le filtre de calibration face-à-face).
   **Preuve empirique** sur `test/turn.mp4` (clip réel, tête+torse) : avant → hull effondré en forme de
   "cône/robe", texture tachetée (`test/out/20260902_164126_webmock/02_result.png`,
   `body=6, rejected=7` sur 24 frames) ; après → buste cohérent tête+épaules
   (`test/out/20260902_164513_after-body-disabled/02_result.png`, `body=0, bodyDetected=19`).
2. ~~**Recentrer le masque par frame**~~ — **Essayé, reverté** : recentrage sur le centroïde brut du masque
   (`maskCenterU`) testé sur `test/turn.mp4`. Résultat pire, pas meilleur (fragment fantôme séparé du corps
   principal, `test/out/20260902_164802_after-recenter/02_result.png`). Cause probable : le centroïde d'un
   masque plein corps varie aussi avec l'**angle** (asymétrie bras/épaules en rotation), pas seulement avec
   une dérive latérale réelle — un centroïde brut mélange donc les deux signaux au lieu d'isoler la dérive.
   Piste correcte si on veut reprendre ça : centrer sur un signal plus stable (bbox du visage via
   `FaceLandmarker`, pas la silhouette entière).
3. ~~**Lissage temporel de l'angle**~~ — **Fait** ([capture-demo.js:50-51](../demo/capture/capture-demo.js#L50-L51),
   `ANGLE_SMOOTH_RADIUS`) : moyenne circulaire sur fenêtre triangulaire ±1, gardée (technique saine, risque
   faible). **Résultat empirique non concluant** : 2 runs sur `test/turn.mp4` toujours dans la même
   fourchette de variance qu'avant (un "cône" propre, un "cône" + fragment) — aucun run n'est pire à cause
   du lissage, mais pas de gain net visible non plus sur ce clip. La vraie cause de variance semble ailleurs
   (voir point suivant).
4. **Valider les constantes de signe** — **Partiellement fait**. `FACE_YAW_SIGN=1` **confirmé correct** :
   filmstrip (`Voir les frames (debug)`) sur `test/turn.mp4` montre une trajectoire d'angle lisse et continue,
   passant par 0° exactement aux frames face-caméra, sans saut de signe — cohérent avec le mouvement réel de
   la tête. `BODY_YAW_SIGN` sans objet (renfort désactivé). `COLMAP_AZIMUTH_SIGN` toujours **non
   vérifiable** sans un vrai run gsplat/COLMAP externe (hors scope, pas de pipeline installé ici).
   **Découverte en cours de route** : `radPerPx` (échelle flux→radians calibrée,
   [capture-demo.js:397-409](../demo/capture/capture-demo.js#L397-L409)) variait d'un facteur **6× d'un run
   à l'autre sur le même clip** (-0.0026 à -0.01686 observés) — la calibration ne reposait que sur la
   médiane de quelques ratios individuels, trop peu d'échantillons, très sensible à la paire précise
   capturée. ~~**Stabilisé**~~ — **Fait** : remplacé par une estimation poolée `sum(dPose)/sum(pxSum)`
   pondérée par le flux de chaque paire plutôt qu'une médiane de ratios individuels, avec un seuil minimum
   (`MIN_CALIBRATION_PAIRS=5`, `MIN_CALIBRATION_FLOW_PX=60`) sous lequel on retombe sur le défaut fixe
   plutôt qu'une valeur calibrée bruitée. **Résultat mesuré** : variance réduite de 6× à ~1,6× sur 4 runs
   (-0.0097 à -0.0156). **Mais** ça n'a **pas** résolu la forme "cône" récurrente (testé sur 2 runs
   post-fix, toujours conique, un avec fragment fantôme) — l'hypothèse "radPerPx = cause dominante" est
   **infirmée**. Nouvelle hypothèse plus probable, cohérente avec une limite déjà documentée juste
   au-dessus (§ [Reconstruction](#comment-ça-marche), point 5) : la forme conique (large en bas, étroit en
   haut) est la signature d'une **profondeur du torse non contrainte** — seules les vues à visage détecté
   ancrent l'angle précisément, rien ne garantit qu'elles couvrent aussi un bon profil des
   **épaules/torse**, qui a besoin de ses propres vues de profil pour contraindre sa profondeur.
4bis. ~~**Corriger la couverture de profil du torse**~~ — **Décision : recadré plutôt que corrigé**. Le
   torse n'étant pas l'objectif de la démo, on a choisi de **couper la zone mal contrainte** plutôt que de
   résoudre sa couverture angulaire (plus gros chantier pour un intérêt marginal ici).
   **Bug trouvé en usage réel et corrigé** : la première version utilisait un `MIN_CARVE_Y` **fixe** en
   espace-monde, tuné sur un seul clip — en pratique, la position de ce seuil dans l'image dépend du
   cadrage webcam réel (distance, hauteur de caméra/siège), donc un utilisateur avec un cadrage différent
   se retrouvait avec **tout coupé** (tête basse dans l'image) ou **tout gardé** (tête haute dans l'image),
   signalé par l'utilisateur après un premier déploiement. **Fix** : le seuil est maintenant **dérivé par
   capture** de la ligne d'épaules mesurée (médiane de `PoseLandmarker.landmarks` 2D sur les frames de la
   capture, convertie en Y-monde via la même relation `v ↔ py` que la boucle de sculpture), avec repli sur
   l'ancienne constante si les épaules ne sont jamais vues avec confiance. Voir
   [`buildPointCloud`](../demo/capture/capture-demo.js) et `yawFromPoseResult`. **Validé** en simulant deux
   cadrages opposés à partir du même clip (sujet décalé haut/bas via crop+pad ffmpeg) : le seuil s'adapte
   correctement dans les deux cas (`carveMinY` -0.428 cadrage bas, 0.113 cadrage haut, contre 0.0 fixe
   avant) et le résultat reste non-vide et cohérent
   (`test/out/20260902_175152_framing-low/`, `..._175303_framing-high/`).
5. **Documenter l'export COLMAP → training 3DGS offline comme voie qualité principale**, le hull temps
   réel restant une démo interactive plutôt qu'un livrable photoréaliste (voir
   [Export COLMAP](#export-colmap-vers-un-pipeline-3dgs-offline)).
6. **Carving perspectif (pinhole)** au lieu d'orthographique dans `buildPointCloud` — gain structurel,
   mais chantier à part (touche le cœur de la reconstruction).

**Essayé et reverté — resserrer `CARVE_KEEP_RATIO`/`MIN_CARVE_VIEWS`** (0.7→0.85, 3→5 vues) pour tuer les
fragments fantômes : deux runs sur le même clip donnent des résultats mitigés (un correct, un très
sur-sculpté en forme de cylindre). Pas un gain net — la variance run-à-run (capture temps réel, non
déterministe) est plus grande que l'effet du réglage ; à ne retenter qu'avec plusieurs runs agrégés, pas un
seul avant/après.

**Audit MediaPipe (vérifié)** : `outputFacialTransformationMatrixes: true` est déjà activé
([capture-demo.js:171](../demo/capture/capture-demo.js#L171)) et le yaw est déjà extrait de cette matrice
4×4, pas d'un `atan2` naïf sur landmarks 2D — c'est déjà la méthode recommandée. `outputFaceBlendshapes`
et un solveur PnP (OpenCV.js) ont été évalués et écartés : le premier mesure l'expression (pas la rigidité
de la pose tête, déjà isolée par la matrice), le second ferait doublon avec un signal déjà fourni par
MediaPipe — aucun des deux n'adresse les causes de bruit observées (torse, texture, seuils de carving).

**Verdict mis à jour (après recadrage tête+cou)** : le torse n'étant pas requis pour cette démo (décision
produit), le recadrage `MIN_CARVE_Y` donne un résultat **nettement plus cohérent et reproductible** que le
buste complet — plus de cône, plus de fragment fantôme sur les runs testés. Le vrai gain qualité pour un
usage non-interactif reste l'export COLMAP → training offline (déjà en place).

*Tous les items prioritaires sont traités. Le hull recadré (tête+cou) reste le mode par défaut ; le mesh
canonique ci-dessous est maintenant un **second mode** sélectionnable, pas un remplacement.*

### Modes mesh / dense / fused — état actuel

Pas d'asset OBJ canonique MediaPipe ni de projection UV (non disponible, aurait ajouté une dépendance
externe) — géométrie = les 478 landmarks bruts (topologie déjà fixe par construction), couleur = fusion
multi-vues par sommet. Voir "Comment ça marche" ci-dessus pour le fonctionnement de chaque mode ; le détail
ci-dessous porte sur les réglages qui répondent à une contrainte précise.

**mesh — rayon de gaussienne adaptatif par landmark.** Un seul rayon global ne peut pas convenir à la fois
aux zones denses (yeux/sourcils → un grand rayon fusionne tout en un flou) et aux zones éparses (joues → un
petit rayon laisse voir le fond entre les points). Le rayon de chaque gaussienne est dérivé de la distance
à son plus proche voisin parmi les 478 landmarks (`nearestNeighborDistances`), borné par
`FACE_MESH_SPLAT_SCALE_MIN/MAX`. `initSceneAndRender` accepte un tableau `perPointScale` optionnel (un
rayon par point) en plus du rayon global partagé — seul le mode mesh l'utilise.

**dense / fused — profondeur d'un vrai modèle monoculaire.** L'interpolation entre les 478 landmarks (IDW)
est une estimation lissée, pas une mesure — insuffisante pour un vrai relief (pas de bosse du nez en
profil). Remplacée par [**Depth Anything v2 (small, q8, ~38 Mo)**](https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX)
via [transformers.js](https://huggingface.co/docs/transformers.js/index), chargé par CDN dans l'`importmap`
de `capture-demo.html` exactement comme `three`/MediaPipe/SparkJS — aucune étape de build, contrainte du
poste (pas de droits admin, `npm install` bloqué par l'EDR). Chargement en parallèle des modèles MediaPipe
mais **non bloquant** : si indisponible (réseau, pas de WASM/WebGPU), repli silencieux sur l'IDW — aucune
régression possible côté disponibilité. La sortie brute du modèle est une profondeur relative d'échelle et
de signe arbitraires ; calibrée par régression linéaire (`fitAffine`) contre le `z` réel des 478 landmarks
(anchors), donc le signe n'a pas besoin d'être connu à l'avance. Inférence faite sur la face **recadrée**
(pas la frame entière) : le budget de résolution du modèle, limité, doit être dépensé sur le visage plutôt
que dilué sur toute l'image pour donner un relief exploitable. `window.__depthStats` (dense) et
`window.__fusionStats` (fused, par vue) exposent la source utilisée (`depth-model`/`idw`) et la calibration.

**fused — alignement multi-vues relatif à la référence, pas absolu.** Chaque vue fusionnée (référence +
jusqu'à `DENSE_FUSION_MAX_VIEWS`=4) a sa propre matrice de rotation complète (`facialTransformationMatrix`,
tangage/roulis inclus, pas juste le lacet) et sa propre calibration de profondeur indépendante — inférences
lancées en parallèle. L'alignement compose la rotation de chaque vue **relativement à la référence**
(`Rdelta = view.faceRot @ transpose(reference.faceRot)`, la référence gardant ses points bruts) plutôt que
de ramener chaque vue à une orientation absolue indépendamment : le bruit de mesure de
`facialTransformationMatrix` se compense bien entre deux frames proches, pas contre une cible absolue.
Deux gardes-fous répondent à la même contrainte sous-jacente — le relief par vue reste modeste (limite du
modèle/IDW ci-dessus), donc une vue complète tournée à grand angle se voit comme un plan plutôt que comme du
volume : `DENSE_FUSION_WING_INNER_FRAC`=0.55 (chaque vue non-référence ne contribue que sa bande extérieure,
pas toute sa bbox, pour ne pas redessiner en double la zone déjà couverte par la référence) et
`DENSE_FUSION_MAX_YAW_DELTA`=30° (au-delà, une vue proche du profil a aussi des landmarks moins fiables — le
relief insuffisant + l'angle important la font apparaître détachée plutôt que rattachée au volume). Les
points de vues différentes tombant dans la même cellule d'une grille (`DENSE_FUSION_MERGE_CELL`=0.010) sont
moyennés plutôt que laissés en doublons.

**Méthode de vérification** : les gaussiennes SparkJS n'ont pas de face cachée (pas de backface culling) —
une géométrie fausse en profondeur peut quand même "ressembler à un visage" sur un screenshot de face, la
couleur seule suffit à la reconnaissance. Toute vérification touchant à la profondeur ou à l'alignement
multi-vues se fait **en tournant la caméra** (`side_view_check.py`, scratchpad de session, réutilisable),
jamais sur un screenshot de face seul.

## Pistes d'évolution (hors scope actuel)
- **Tester `DEPTH_MODEL_DTYPE="q4f16"`** (~20 Mo au lieu de `q8` ~38 Mo) — poids plus léger, à valider que la
  qualité du relief reste acceptable avant de remplacer le défaut actuel.

### EN COURS — mode fused encore "brouillon" (points dédoublés/fantômes)
Signalé après le passage en anisotrope : le mode fused affichait des points dédoublés/mal alignés,
nettement pire que dense (confirmé par l'utilisateur en webcam réelle **et** sur les screenshots).
Deux pistes testées et **mesurées comme fausses** avant d'en trouver une vraie — gardé ici plutôt
qu'effacé, l'historique évite de retester une piste déjà invalidée :
- ~~Densité de grille des vues secondaires (`DENSE_FUSION_STEP_MULT` 2→1)~~ — améliore la texture
  mais ne change rien au symptôme réel signalé par l'utilisateur ("pas vraiment mieux").
- ~~Rejet par qualité de calibration profondeur (`r2` de `fitAffine`)~~ — mesuré sur 4 captures :
  r2 ~0.42-0.66 pour **toutes** les vues y compris la référence (qui fonctionne bien en mode dense
  seul). Le r2 ne distingue pas une bonne vue d'une mauvaise ici, seuil supprimé
  (`MIN_DEPTH_CALIB_R2 = -Infinity`).
- ~~Distance au landmark le plus proche par pixel (`DEPTH_MODEL_MAX_LANDMARK_DIST`)~~ — implémenté
  et gardé (correct sur le principe : ne pas faire confiance à une extrapolation du modèle de
  profondeur loin de tout ancrage réel), mais **mesuré sans effet sur le fantôme observé** — gardé
  car défendable seul, pas comme correctif.

**Root cause identifiée** (diagnostic par coloration des points par vue source,
`window.__fusedDebugColorByView`, scratchpad de session) : la vue la plus éloignée dans une fusion
se détache visuellement en une surface distincte — pas du bruit par pixel, un décalage de rotation
systématique sur toute la vue. Cohérent avec la fiabilité connue du head-pose MediaPipe qui se
dégrade près du profil (voir Limites connues, ~±70°). Mesuré directement via
[`viewAlignError`](../demo/capture/capture-demo.js) : transforme les 478 landmarks de la vue par
`Rdelta` et compare à ceux de la référence au même index (topologie fixe = vraie correspondance,
pas une approximation) — une vue proche de la référence mesure ~0.002, les vues qui se détachent
visuellement ~0.006-0.008.

**Statut : calibré et validé par mesure objective**, pas seulement à l'œil. Un premier seuil
(`0.004`) rejetait systématiquement les 4 vues secondaires (fused dégénérait en dense seul). Un
outil dédié — [`test/reconstruction_quality.py`](../test/reconstruction_quality.py), demandé par
l'utilisateur après avoir refusé un jugement à l'œil sur des screenshots — calcule maintenant deux
scores objectifs à partir du nuage de points réellement rendu (`window.__lastPoints`) :
- **cohérence géométrique** : fraction des points dans la plus grande composante spatialement
  connexe (union-find sur un hash spatial) — un fantôme séparé fait baisser ce score.
- **cohérence photométrique** (fused uniquement) : variance de couleur moyenne dans les cellules
  de la grille de fusion qui ont reçu des contributions de **plusieurs** vues
  (`window.__fusionColorVariance`, champs `r2/g2/b2` ajoutés à l'accumulateur existant) — si deux
  vues voient vraiment le même point physique, leurs couleurs échantillonnées doivent se
  ressembler ; sinon c'est un second signal de désalignement, indépendant du premier.

Mesure (3 runs, garde-fou désactivé) : `alignErr` non-référence regroupé 0.003-0.0095, sans
séparation nette bonne/mauvaise vue, et la cohérence géométrique de fused (0.95-0.98) restait
proche de sa propre base dense/mesh même avec **toutes** les vues incluses — le fantôme visible à
l'œil sur une capture n'était pas représentatif de la moyenne. Seuil retenu : **`0.008`**
(au-dessus de la plage typique, ne coupe que les pires cas), revalidé sur 3 runs supplémentaires :
fusion multi-vues réelle toujours présente (2-3 vues secondaires survivent, ~2-2.5× les points de
dense) **et** variance photométrique mesurablement meilleure (0.006-0.007 contre 0.009-0.028 garde-
fou désactivé) — premier changement de cette investigation avec une amélioration réellement
mesurée, pas seulement supposée.

- **Pondération de confiance par vue en mode fused** au lieu d'un rejet tout-ou-rien : pondérer la
  contribution d'une vue par sa qualité d'alignement (`alignErr`) plutôt que de l'inclure/exclure
  entièrement pourrait mieux utiliser les vues "moyennes" au lieu de les jeter — piste restante,
  `reconstruction_quality.py` permettrait de la valider objectivement comme pour le seuil ci-dessus.
- **Entraîner un vrai 3DGS offline** à partir de l'export COLMAP (gsplat/Brush sur GPU), puis réimport du
  `.sog` entraîné dans le viewer SparkJS — le « vrai » 3DGS, la démo servant de front-end de capture.
  **Statut : non commencé**, piste [Brush](https://github.com/ArthurBrussee/brush) déjà évaluée (voir
  section précédente)
- Fusion multi-passes pour densifier le volume

## Fichiers
- [`capture-demo.html`](../demo/capture/capture-demo.html) — UI (webcam, progression, overlay stats, résultat)
- [`capture-demo.js`](../demo/capture/capture-demo.js) — pipeline complet : capture, segmentation, reconstruction, rendu

## Test automatisé

Un harnais [`test/run_capture_test.py`](../test/run_capture_test.py) pilote le Chrome installé en headless
avec une **fausse webcam** (`--use-file-for-fake-video-capture`), permettant de valider le pipeline sans
caméra physique. Dépendances : `pip install playwright imageio-ffmpeg` (réutilise Chrome via
`channel="chrome"`, pas de téléchargement de navigateur).

```
python test/run_capture_test.py                       # synthétique (valide l'infra ; 0 frame capturée)
python test/run_capture_test.py --clip test/turn.mp4   # test complet avec un vrai clip
python test/run_capture_test.py --label before-fix     # tague le sous-dossier de sortie
```

Le test complet exige un clip d'une **vraie personne** (`test/turn.mp4`) : le `PoseLandmarker` ne retient
une frame que si une personne est détectée. Chaque run écrit ses captures dans un **sous-dossier
horodaté** (`test/out/<horodatage>[_label]/01_ready.png`, `02_result.png`, `03_mesh.png`,
`04_back_to_hull.png` si le mode mesh est disponible) au lieu d'écraser le run précédent — pratique pour
comparer avant/après un correctif du [TODO](#todo).

> Trois modèles MediaPipe se chargent (segmentation, pose, visage). L'angle vient en priorité du **visage**
> (l'ancre), sinon du **flux optique** — le renfort par pose du torse est désactivé par défaut
> (`BODY_YAW_REINFORCE`, voir [TODO](#todo)). La sonde `window.__angleStats` (imprimée par le harnais)
> indique la répartition — p. ex. `face=12, flow=8, body=0`. `window.__carveStats` indique où le
> recadrage tête+cou dynamique a atterri pour ce run.

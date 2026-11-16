# Audit comparatif EVA 2023 / Végéstrate 2023

## Synthèse

Deux inventaires de la végétation 2023 de la Métropole de Lyon sont comparés sur $domaine_km2 km² : **EVA 2023**, photo-interprétation manuelle en polygones, et **Végéstrate 2023**, raster à 0,2 m issu du LiDAR et des orthophotographies qui est ensuite vectorisé. **L'audit mesure l'accord entre les deux sources** : aucune n'est une vérité terrain. Les pourcentages portent sur la surface comparée ; sauf mention contraire, ils sont calculés au pixel de 0,2 m, EVA étant rastérisée sur la grille Végéstrate.

- **Végétal / non végétal : $accord_vegetation d'accord.** Les deux sources voient la végétation aux mêmes endroits.
- **Ligneux / herbacé : $accord_ligneux d'accord ; arbre / arbuste / herbe : $accord_strates.** Plus la comparaison entre dans le détail des strates, plus les définitions diffèrent (arbuste dès 1 m pour EVA, végétation moyenne dès 1,5 m pour la classification LiDAR utilisée par Végéstrate ; ce seuil peut être adapté au besoin, en reclassant le LiDAR brut ou en seuillant directement le modèle de hauteur).
- **Forêts (EVA code 311) : Végéstrate retrouve la couverture ligneuse attendue sur $conforme_311 de leur surface**, à l'échelle du polygone EVA.
- **Le désaccord est concentré sur les contours.** À l'intérieur des zones végétales EVA, Végéstrate voit de la végétation sur $bord_dans_pres des pixels à moins de $bord_pres m du bord et sur $bord_dans_loin à plus de $bord_loin m.
- **Végéstrate voit la végétation que l'unité minimale d'EVA (100 m²) écarte** : $taches_ligneux_non_vegetal de la végétation ligneuse Végéstrate située en zone EVA non végétale est en taches de moins de 100 m² (arbres isolés, alignements), soit $taches_ligneux_non_vegetal_km2 km².

![Cinq lieux d'exemple : orthophotographie, EVA 2023, Végéstrate 2023 et accord végétation](figures/lieux.jpg)

![Part des pixels vus végétaux (ou ligneux) par Végéstrate selon la distance au bord des zones EVA](figures/distance_bord.png)

## Données

| Donnée | Description |
|---|---|
| EVA 2023 | Produit par Geofit pour la Métropole par photo-interprétation manuelle. Polygones, champs `COD3` (classe) et `CODH` (classe de hauteur), EPSG:3946. Unité minimale de collecte (UMC) 100 m², largeur minimale de collecte (LMC) 3 m, taux de couverture propre à chaque classe. |
| Végéstrate 2023, classes | Produit par TelesCoop et le LIRIS. Raster 0,2 m, EPSG:3946. 0 autre, 1 herbacé (FLAIR-HUB), 2 végétation moyenne (LiDAR 1,5-5 m), 3 végétation haute (LiDAR > 5 m). Les zones « Autre » du LiDAR reconnues comme végétation moyenne ou haute par FLAIR-HUB sont complétées (`src/postprocessing/merge_classifications.py`), puis un filtre modal 3x3 est appliqué pour lisser. |
| IRIS, communes, LCZ | Base de données data.gouv (IRIS INSEE, zones climatiques locales). |

Le domaine est l'union des IRIS de la Métropole : les deux sources y sont découpées.

**Zone exclue.** L'orthophotographie 2023 est vide sur le camp militaire du Mont Verdun : Végéstrate n'y a pas de donnée. EVA a complété cette zone grâce à des sources secondaires (OSM, IGN, etc.). Le contour est tracé à partir des pixels transparents de l'orthophotographie Grand Lyon 2023 (WMS `ortho_2023`, 0,5 m). Ces $exclusion_km2 km² sont retirés pour les deux sources ; leur contenu EVA est dans `zone_exclue.csv`.

![Zone exclue](figures/zone_exclue.jpg)

## Harmonisation des classes

EVA mêle des classes de couverture et des classes d'usage du sol ; Végéstrate ne décrit que la couverture physique. Chaque classe EVA est rattachée à un groupe de couverture attendue :

$groupes

Les classes « non comparable » (cultures classées à partir du RPG, friches, jardins collectifs, vergers) sont laissées hors des matrices. Comptées comme végétales, l'accord végétal / non végétal passe à $accord_vegetation_nc.

## Recoupage des hauteurs

EVA distingue < 1 m, 1-3 m et > 3 m ; Végéstrate distingue la végétation moyenne (1,5-5 m) et haute (> 5 m). Pour comparer les strates, Végéstrate est recoupé sur les seuils d'EVA (variante principale : $coupe_principale, soit arbuste dès $arbuste_min m arrondi et arbre dès $arbre_min m arrondi) :

- les pixels herbacés (classe 1, FLAIR-HUB) restent de l'herbe ;
- les zones de végétation moyenne ou haute (classes 2 et 3) sont reclassées selon leur hauteur médiane ;
- une zone de classe 2 ou 3 dont la hauteur médiane est nulle n'a pas de hauteur mesurée : elle garde sa classe Végéstrate native. C'est le cas de la végétation reconnue par FLAIR-HUB là où le LiDAR ne donne pas de hauteur, notamment autour du camp militaire du Mont Verdun, où la lacune du LiDAR semble plus large que celle de l'orthophotographie.

Une zone est un ensemble 4-connexe de pixels de même classe, la règle utilisée pour vectoriser Végéstrate. La hauteur d'une zone est la hauteur médiane sur ses pixels, comme l'attribut `median_height` des polygones Végéstrate publiés dans IA.rbre (`compute_zone_elevation` dans `src/postprocessing/apply_vegetation_elevation.py`). Les zones sont calculées par bloc de 2048 pixels avec une marge de 20 m : un massif plus grand prend la médiane de sa partie visible.

Un pixel herbacé de Végéstrate est un pixel où le LiDAR n'a pas vu de végétation et où FLAIR-HUB a reconnu de l'herbe.

Le modèle de hauteur de végétation de Végéstrate est arrondi au mètre (paramétrisable), chaque seuil n'est donc connu qu'à ± 0,5 m. Selon la variante de seuils ($variantes), l'accord par strate varie de $strates_min à $strates_max ; le détail est dans `indicateurs_accord.csv`.

## Niveaux, échelles et indicateurs

La comparaison se fait sur trois niveaux : **végétation** (végétal / non végétal), **ligneux (strates hautes et moyennes) / herbacé**, **strates** (arbre / arbuste / herbe, sur les seules classes EVA à strate unique : forêts, haies et fourrés sont exclus de ce niveau).

Les deux données ont des échelles différentes : **le pixel** de 0,2 m pour Végéstrate (qui est ensuite vectorisé) et **le polygone pour EVA**.

Les indicateurs sélectionnés pour les comparaisons sont :

- **Accord global** et **décomposition du désaccord** : désaccord de **quantité** (surfaces totales différentes) et d'**allocation** (surfaces semblables à des endroits différents).
- **Composition des polygones EVA** : part des polygones EVA (et de leur surface) où Végéstrate retrouve la couverture attendue par la définition de leur classe (taux de couverture du guide EVA). C'est la comparaison à l'échelle d'EVA.
- **Accord selon la distance au bord** des zones EVA, dedans et dehors.
- **Petites taches** : végétation Végéstrate en taches 4-connexes de moins de 100 m² (l'UMC d'EVA).
- **Couverture par commune**.

## Résultats

### Accord global

$indicateurs

### Matrices de confusion

Répartition de chaque classe EVA (ligne) entre les classes Végéstrate (colonne) ; la diagonale correspond à l'accord.

![Niveau végétation](figures/matrice_vegetation.png) ![Niveau ligneux / herbacé](figures/matrice_ligneux.png) ![Niveau strates](figures/matrice_strates.png)

### À l'échelle des polygones EVA

Pour chaque classe EVA, part des polygones (et de leur surface) où Végéstrate retrouve la couverture attendue par la définition de la classe (« ligneux » = arbre + arbuste), variante principale. Végéstrate retrouve cette couverture sur $conforme_311 de la surface des forêts et $conforme_111 de la strate arborée, mais sur $conforme_112 seulement de la strate arbustive, ce qui est cohérent avec la différence de définition de l'arbuste (dès 1 m pour EVA, 1,5 m pour la classification LiDAR). Détail par classe dans `regles_couverture.csv`.

![Couverture attendue retrouvée par Végéstrate dans les polygones EVA](figures/regles_couverture.png)

### Accord selon la distance au bord des zones EVA

Voir la figure de la synthèse. Dans les zones EVA végétales, Végéstrate voit de la végétation sur $bord_dans_pres des pixels à moins de $bord_pres m du bord, $bord_dans_loin à plus de $bord_loin m. Hors des zones EVA végétales, il en voit sur $bord_hors_pres des pixels à moins de $bord_pres m du bord, $bord_hors_loin à plus de $bord_loin m. Le désaccord se concentre dans les premiers mètres autour des contours.

### Végétation en petites taches (< 100 m²)

Sur toute la végétation Végéstrate, $taches_vegetation est en taches de moins de 100 m². Dans les zones EVA non végétales, cette part monte à $taches_ligneux_non_vegetal pour la végétation ligneuse : arbres isolés et alignements qu'EVA n'a pas à cartographier.

![Petites taches](figures/petites_taches.png)

### Couverture par commune

Parts calculées sur la surface comparable de chaque commune, une commune par point, taille proportionnelle à la surface. Détail dans `couverture_communes.csv` et `couverture_iris.csv`.

![Couverture par commune](figures/communes.png)

La correspondance est globalement la même pour toutes les communes : aucune ne se démarque nettement.

## Limites

- Accord n'est pas exactitude : sans vérité terrain, un désaccord peut venir de l'une, de l'autre ou des deux sources.
- EVA cartographie la couverture dominante de polygones d'au moins 100 m² avec un taux de couverture de 60 à 75 % : jusqu'à un quart d'un polygone « strate arborée » peut légitimement être autre chose, et ses contours englobent la végétation. Au pixel, ces écarts admis par la définition d'EVA comptent comme des désaccords ; la composition des polygones EVA donne l'accord à l'échelle d'EVA.
- Le modèle de hauteur de canopée (nDSM Végéstrate) est arrondi au mètre et ne porte que sur les pixels que Végéstrate classe en végétation : là où EVA voit un arbre et Végéstrate rien, aucune hauteur ne permet de distinguer un écart de définition d'un écart de détection.

## Reproduire

Ce document est généré à partir de `audit/methodology_template.md` et de `audit/config.yaml` ; les chiffres proviennent des tableaux CSV produits dans le dossier de sortie de l'audit.

```bash
python -m audit.run --config audit/config.yaml
```

Étapes : `audit.prepare` (zone exclue, export IRIS et LCZ depuis la base IA.rbre), `audit.compute` (comptages pixel par blocs), `audit.examples` (rasters des lieux d'exemple), `audit.report` (tableaux et ce document), `audit.figures` (figures), `audit.publish` (PDF et copies vers la documentation et le site). `python -m audit.selfcheck` vérifie les fonctions de calcul sur des cas simples.

Tableaux complets dans le dossier de sortie :

- `indicateurs_accord.csv` : accord et désaccord pour tous les niveaux et toutes les variantes de seuils ;
- `matrices.csv` : matrices en km², parts par ligne et par colonne ;
- `regles_couverture.csv`, `polygones_composition.csv.gz` : couverture attendue retrouvée par classe et composition de chaque polygone ;
- `distance_bord.csv`, `petites_taches.csv` ;
- `couverture_communes.csv`, `couverture_iris.csv`, `accord_lcz.csv` ;
- `zone_exclue.csv`, `exemples_stats.csv` et les rasters `exemples/<lieu>_*.tif` avec leurs styles QGIS `.qml`.

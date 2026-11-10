# Audit comparatif EVA 2023 / Vegestrate 2023

Ce document est généré par `python -m audit.run` à partir de `audit/methodology_template.md` et de `audit/config.yaml`. Les chiffres proviennent des tableaux CSV produits dans le dossier de sortie de l'audit.

## Objet

Comparer, sur toute la Métropole de Lyon, deux inventaires de la végétation produits à partir des données 2023 :

- **EVA 2023** (Espaces végétalisés et artificialisés), produit par Geofit pour la Métropole par photo-interprétation manuelle, en polygones ;
- **Vegestrate 2023**, produit par TelesCoop et le LIRIS par fusion de la classification LiDAR 2023 et de la segmentation des orthophotographies 2023 par le modèle FLAIR-HUB de l'IGN, en raster à 0,2 m.

**Cet audit mesure l'accord entre deux jeux de données, pas leur exactitude.** Aucune des deux sources n'est une vérité terrain et aucune nouvelle annotation n'a été réalisée. Quand les deux sources divergent, l'audit ne dit pas laquelle a raison : il décrit où, combien et dans quelles situations elles divergent. Les tableaux sont donnés dans les deux sens (EVA en référence, puis Vegestrate en référence) pour qu'aucune des deux ne serve d'arbitre.

## Données

| Donnée | Description |
|---|---|
| EVA 2023 | Polygones, champ `COD3` (classe) et `CODH` (classe de hauteur), EPSG:3946. Unité minimale de collecte (UMC) 100 m², largeur minimale de collecte (LMC) 3 m, taux de couverture propre à chaque classe (guide de photo-interprétation Geofit). |
| Vegestrate 2023, classes | Raster 0,2 m, EPSG:3946. 0 autre, 1 herbacé (FLAIR-HUB), 2 végétation moyenne (LiDAR 1,5-5 m), 3 végétation haute (LiDAR > 5 m). Les zones « Autre » du LiDAR reconnues comme végétation moyenne ou haute par FLAIR-HUB sont complétées (`src/postprocessing/merge_classifications.py`), puis un filtre modal 3x3 est appliqué. |
| Vegestrate 2023, hauteurs | Modèle numérique de hauteur (nDSM) à 0,2 m sur les seuls pixels de végétation, hauteur maximale du pixel arrondie au mètre (`src/core/utils.py`). |
| IRIS, communes, LCZ | Base de données IA.rbre (IRIS INSEE, zones climatiques locales). |

## Domaine d'analyse et zone exclue

Le domaine est l'union des IRIS de la Métropole, soit $domaine_km2 km² analysés après exclusion. EVA dépasse le territoire métropolitain et Vegestrate couvre un rectangle plus large : les deux sont découpés sur ce domaine.

**Zone exclue.** L'orthophotographie 2023 est vide (pixels transparents) sur le camp militaire du Mont Verdun (Monts d'Or). Vegestrate n'y contient donc aucune donnée, alors qu'EVA 2023 y est cartographiée. Le contour exact est tracé à partir des pixels transparents de l'orthophotographie Grand Lyon 2023 (service WMS `ortho_2023`, résolution 0,5 m), et non à partir de l'enceinte du camp. Cette zone, soit $exclusion_km2 km² dans le domaine, est retirée de toutes les statistiques pour les deux sources. Son contenu est donné à part :

$exclusion

Surface du domaine sans polygone EVA : $eva_absente_km2 km².

## Harmonisation des classes

EVA mêle des classes de couverture (strates arborée, arbustive, herbacée) et des classes d'usage du sol (cultures, prairies, forêts, jardins collectifs). Vegestrate ne décrit que la couverture physique. Chaque classe EVA est rattachée à un groupe de couverture attendue :

$groupes

Les classes du groupe « non comparable » ne sont pas forcées dans les matrices : cultures dont la couverture dépend de la saison et qui sont classées à partir du RPG (Registre parcellaire graphique), friches, jardins collectifs et vergers aux strates mêlées par définition. Leur surface est indiquée comme « hors matrice ». Un niveau de sensibilité (`vegetation_avec_non_comparable`) les compte comme végétales.

## Recoupage des hauteurs

Les seuils de hauteur diffèrent : EVA distingue < 1 m, 1-3 m et > 3 m ; Vegestrate distingue la végétation moyenne (1,5-5 m) et haute (> 5 m). Une végétation de 4 m est un arbre pour EVA et une végétation moyenne pour Vegestrate. Pour le niveau « strates », Vegestrate est donc aussi présenté recoupé sur les seuils d'EVA avec le nDSM :

- les pixels herbacés (classe 1, FLAIR-HUB) restent de l'herbe ;
- les zones de végétation moyenne ou haute (classes 2 et 3) sont reclassées selon leur hauteur médiane.

Une zone est un ensemble 4-connexe de pixels de même classe, c'est-à-dire un polygone de la version vectorielle de Vegestrate (`src/postprocessing/vectorize_raster.py`). Sa hauteur est la médiane du nDSM sur ses pixels, tronquée au mètre, comme dans la phase `zone_elevation` du pipeline Vegestrate (`compute_zone_elevation` dans `src/postprocessing/apply_vegetation_elevation.py`) et comme l'attribut `median_height` des polygones Vegestrate publiés dans IA.rbre. Un recoupage pixel par pixel classerait en arbuste, voire en herbe, le bord bas des houppiers et les branches basses, alors que le polygone Vegestrate correspondant est un arbre. Le recoupage pixel par pixel est conservé comme option (`height_source: pixel` dans `audit/config.yaml`). Les zones sont calculées par fenêtre de calcul (bloc de 2048 pixels et marge de 20 m) : une zone plus grande que la fenêtre, typiquement un massif forestier, prend la moyenne de sa partie visible.

Ce choix a une conséquence connue : un pixel herbacé situé sous un houppier reste de l'herbe, alors qu'EVA, qui interprète une vue du dessus, le classe en strate arborée. Ce désaccord pèse donc contre Vegestrate.

Le nDSM est arrondi au mètre : chaque seuil n'est connu qu'à ± 0,5 m près. Trois variantes sont calculées :

$coupes

La version « natif » correspond aux classes Vegestrate sans recoupage (herbe = classe 1, arbuste = classe 2, arbre = classe 3).

## Niveaux et échelles de comparaison

Trois niveaux, du plus au moins aligné entre les définitions :

1. **végétation** : végétal / non végétal ;
2. **ligneux_herbace** : ligneux / herbacé / non végétal ;
3. **strates** : arbre / arbuste / herbe / non végétal, sur les seules classes EVA à strate unique (les forêts, haies et fourrés mêlent plusieurs strates par définition et sont exclus de ce niveau).

Deux échelles :

- **le pixel** de 0,2 m, EVA étant rastérisé sur la grille Vegestrate (un pixel prend la valeur du polygone qui contient son centre) ;
- **le polygone EVA**, l'échelle pour laquelle EVA a été conçue : la composition Vegestrate de chaque polygone est confrontée au taux de couverture que le guide fixe pour sa classe.

## Indicateurs

- **Matrices de confusion** en km², normalisées par ligne (part de chaque classe EVA) et par colonne (part de chaque classe Vegestrate).
- **Accord global** et **décomposition du désaccord** selon Pontius et Millones (2011) : le désaccord de **quantité** vient de surfaces totales différentes, le désaccord d'**allocation** vient de surfaces semblables placées à des endroits différents. Le kappa de Cohen est donné pour mémoire.
- **Conformité des polygones** : part des polygones EVA (et de leur surface) dont la composition Vegestrate respecte la règle de couverture du guide. Seuls les polygones dont au moins la moitié de la surface est dans le domaine sont retenus.
- **Accord selon la distance au bord** des zones EVA (dedans et dehors), pour mesurer la part du désaccord liée au tracé des contours.
- **Petites taches** : surface de végétation Vegestrate dans des taches de moins de 100 m² (l'UMC d'EVA), c'est-à-dire en dessous de ce qu'EVA cartographie par construction. Les taches sont des composantes 4-connexes. Une tache qui touche le bord de la fenêtre de calcul (marge de 20 m) est comptée comme grande, ce qui sous-estime légèrement les petites taches.
- **Couverture par commune et par IRIS**, calculée pour les deux sources sur la même surface comparable.
- **Accord par type de tissu urbain** (zones climatiques locales, LCZ).

## Résultats

### Accord global

$indicateurs

### Niveau végétation

$matrices_vegetation

### Niveau végétation, classes non comparables comptées comme végétales

$matrices_vegetation_avec_non_comparable

### Niveau ligneux / herbacé

$matrices_ligneux_herbace

### Niveau strates

$matrices_strates

### Conformité des polygones EVA à leur règle de couverture

La règle porte sur la part de la surface du polygone occupée par la strate cible selon Vegestrate (« ligneux » = arbre + arbuste).

$regles

### Accord selon la distance au bord des zones EVA

« dans EVA » : pixels à l'intérieur d'une zone EVA végétale (ou ligneuse), part que Vegestrate voit aussi végétale (ou ligneuse). « hors EVA » : pixels EVA non végétaux (ou non ligneux), part que Vegestrate voit végétale (ou ligneuse). La distance est mesurée au pixel le plus proche de l'autre côté du contour.

$bords

### Végétation en petites taches (< 100 m²)

$taches

### Couverture par commune

Parts calculées sur la surface comparable de chaque commune (classes EVA non comparables et zone exclue retirées pour les deux sources). Le détail par IRIS est dans `couverture_iris.csv`.

$communes

### Accord par zone climatique locale (LCZ)

$lcz

### Lieux d'exemple

Pour chaque lieu (1 km²), les rasters `exemples/<lieu>_*.tif` et leurs styles QGIS `.qml` : groupes EVA, Vegestrate natif, Vegestrate recoupé, accord végétation et accord par strate. Répartition des catégories d'accord par strate :

$exemples

## Limites

- Accord n'est pas exactitude : sans vérité terrain, un désaccord peut venir de l'une, de l'autre ou des deux sources.
- EVA cartographie la couverture dominante de polygones d'au moins 100 m² avec un taux de couverture de 60 à 75 % : jusqu'à un quart d'un polygone « strate arborée » peut légitimement être autre chose. La comparaison au pixel pénalise EVA pour un détail qu'elle ne cherche pas à représenter, d'où la comparaison à l'échelle du polygone.
- Le guide EVA n'indique ni précision de positionnement, ni date ou source des images interprétées.
- Le nDSM est arrondi au mètre et ne porte que sur les pixels que Vegestrate classe en végétation : là où EVA voit un arbre et Vegestrate rien, aucune hauteur ne permet de distinguer un écart de définition d'un écart de détection.
- Les LCZ sont elles-mêmes dérivées en partie de la végétation ; l'accord par LCZ décrit des contextes, pas des causes.

## Reproduire

```bash
python -m audit.run --config audit/config.yaml
```

Étapes : `audit.prepare` (zone exclue, export IRIS et LCZ depuis la base IA.rbre), `audit.compute` (comptages pixel par blocs), `audit.examples` (rasters des lieux d'exemple), `audit.report` (tableaux et ce document). `python -m audit.selfcheck` vérifie les fonctions de calcul sur des cas simples.

## Sources

- Geofit, *Guide de photo-interprétation des données EVA*, Métropole de Lyon, version 1 (pages citées dans `audit/config.yaml`).
- Pontius R. G. Jr., Millones M. (2011), *Death to Kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment*, International Journal of Remote Sensing, 32(15), 4407-4429.
- Vegestrate : ce dépôt, `README.md` et code cité ci-dessus.

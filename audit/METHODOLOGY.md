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

Le domaine est l'union des IRIS de la Métropole, soit 527,964 km² analysés après exclusion. EVA dépasse le territoire métropolitain et Vegestrate couvre un rectangle plus large : les deux sont découpés sur ce domaine.

**Zone exclue.** L'orthophotographie 2023 est vide (pixels transparents) sur le camp militaire du Mont Verdun (Monts d'Or). Vegestrate n'y contient donc aucune donnée, alors qu'EVA 2023 y est cartographiée. Le contour exact est tracé à partir des pixels transparents de l'orthophotographie Grand Lyon 2023 (service WMS `ortho_2023`, résolution 0,5 m), et non à partir de l'enceinte du camp. Cette zone, soit 5,865 km² dans le domaine, est retirée de toutes les statistiques pour les deux sources. Son contenu est donné à part :

| source | classe | surface_km2 |
|---|---|---|
| EVA | arbore | 0,087 |
| EVA | arbustif | 0,096 |
| EVA | boise_mixte | 3,320 |
| EVA | herbace | 1,106 |
| EVA | non_comparable | 0,855 |
| EVA | non_vegetal | 0,401 |
| Vegestrate | arbre | 0,000 |
| Vegestrate | arbuste | 0,000 |
| Vegestrate | aucune | 5,865 |
| Vegestrate | herbe | 0,000 |

Surface du domaine sans polygone EVA : 0,000 km².

## Harmonisation des classes

EVA mêle des classes de couverture (strates arborée, arbustive, herbacée) et des classes d'usage du sol (cultures, prairies, forêts, jardins collectifs). Vegestrate ne décrit que la couverture physique. Chaque classe EVA est rattachée à un groupe de couverture attendue :

| groupe | libellé | codes EVA (page du guide) |
|---|---|---|
| arbore | Strate arborée (> 3 m) | 111 (p. 3) |
| arbustif | Strate arbustive (1-3 m) | 112 (p. 4) |
| herbace | Herbacé | 113 (p. 5), 221 (p. 15), 222 (p. 16), 223 (p. 17), 224 (p. 18), 225 (p. 19), 361 (p. 30), 362 (p. 31), 411 (p. 37) |
| boise_mixte | Boisé multistrate (forêts, haies, fourrés) | 241 (p. 21), 311 (p. 22), 312 (p. 23), 321 (p. 24), 322 (p. 25), 331 (p. 26), 341 (p. 27), 342 (p. 28), 351 (p. 29), 371 (p. 32), 372 (p. 33), 374 (p. 34) |
| non_vegetal | Non végétalisé | 121 (p. 8), 122 (p. 9), 381 (p. 35), 382 (p. 36) |
| eau | Eau | 511 (p. 39), 521 (p. 40), 522 (p. 41), 523 (p. 42), 524 (p. 43), 525 (p. 44), 531 (p. 45), 532 (p. 46), 533 (p. 47) |
| non_comparable | Non comparable (usage du sol, strates mêlées, cultures) | 114 (p. 6), 115 (p. 7), 211 (p. 10), 212 (p. 11), 213 (p. 12), 214 (p. 13), 215 (p. 14), 231 (p. 20), 412 (p. 38) |

Les classes du groupe « non comparable » ne sont pas forcées dans les matrices : cultures dont la couverture dépend de la saison et qui sont classées à partir du RPG (Registre parcellaire graphique), friches, jardins collectifs et vergers aux strates mêlées par définition. Leur surface est indiquée comme « hors matrice ». Un niveau de sensibilité (`vegetation_avec_non_comparable`) les compte comme végétales.

## Recoupage des hauteurs

Les seuils de hauteur diffèrent : EVA distingue < 1 m, 1-3 m et > 3 m ; Vegestrate distingue la végétation moyenne (1,5-5 m) et haute (> 5 m). Une végétation de 4 m est un arbre pour EVA et une végétation moyenne pour Vegestrate. Pour le niveau « strates », Vegestrate est donc aussi présenté recoupé sur les seuils d'EVA avec le nDSM :

- les pixels herbacés (classe 1, FLAIR-HUB) restent de l'herbe ;
- les zones de végétation moyenne ou haute (classes 2 et 3) sont reclassées selon leur hauteur médiane.

Une zone est un ensemble 4-connexe de pixels de même classe, c'est-à-dire un polygone de la version vectorielle de Vegestrate (`src/postprocessing/vectorize_raster.py`). Sa hauteur est la médiane du nDSM sur ses pixels, tronquée au mètre, comme dans la phase `zone_elevation` du pipeline Vegestrate (`compute_zone_elevation` dans `src/postprocessing/apply_vegetation_elevation.py`) et comme l'attribut `median_height` des polygones Vegestrate publiés dans IA.rbre. Un recoupage pixel par pixel classerait en arbuste, voire en herbe, le bord bas des houppiers et les branches basses, alors que le polygone Vegestrate correspondant est un arbre. Le recoupage pixel par pixel est conservé comme option (`height_source: pixel` dans `audit/config.yaml`). Les zones sont calculées par fenêtre de calcul (bloc de 2048 pixels et marge de 20 m) : une zone plus grande que la fenêtre, typiquement un massif forestier, prend la moyenne de sa partie visible.

Ce choix a une conséquence connue : un pixel herbacé situé sous un houppier reste de l'herbe, alors qu'EVA, qui interprète une vue du dessus, le classe en strate arborée. Ce désaccord pèse donc contre Vegestrate.

Le nDSM est arrondi au mètre : chaque seuil n'est connu qu'à ± 0,5 m près. Trois variantes sont calculées :

| variante | arbuste dès (m arrondi) | arbre dès (m arrondi) | description |
|---|---|---|---|
| principal | 1 | 4 | lecture littérale sur hauteurs arrondies : < 1 m herbe, 1 à 3 m arbuste, > 3 m arbre |
| seuils_bas | 1 | 3 | sensibilité : arbre dès 3 m arrondi (hauteur réelle >= 2,5 m) |
| seuils_hauts | 2 | 4 | sensibilité : arbuste dès 2 m arrondi (hauteur réelle >= 1,5 m) |

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

| niveau | version_vegestrate | surface_km2 | accord | desaccord_quantite | desaccord_allocation | kappa |
|---|---|---|---|---|---|---|
| vegetation | natif | 445,823 | 92,5 % | 0,9 % | 6,6 % | 0,848 |
| vegetation_avec_non_comparable | natif | 527,964 | 93,4 % | 0,5 % | 6,1 % | 0,858 |
| ligneux_herbace | natif | 445,823 | 84,6 % | 3,0 % | 12,4 % | 0,762 |
| ligneux_herbace | principal | 445,823 | 84,4 % | 4,6 % | 11,0 % | 0,759 |
| ligneux_herbace | seuils_bas | 445,823 | 84,4 % | 4,6 % | 11,0 % | 0,759 |
| ligneux_herbace | seuils_hauts | 445,823 | 84,0 % | 5,7 % | 10,3 % | 0,753 |
| strates | natif | 380,897 | 80,0 % | 3,0 % | 17,0 % | 0,690 |
| strates | principal | 380,897 | 80,5 % | 5,5 % | 13,9 % | 0,697 |
| strates | seuils_bas | 380,897 | 79,7 % | 7,7 % | 12,6 % | 0,684 |
| strates | seuils_hauts | 380,897 | 80,4 % | 6,6 % | 13,0 % | 0,694 |

### Niveau végétation

**Version Vegestrate : natif**

| EVA \ Vegestrate (km²) | vegetal | non_vegetal |
|---|---|---|
| vegetal | 234,599 | 14,621 |
| non_vegetal | 18,764 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | vegetal | non_vegetal |
|---|---|---|
| vegetal | 94,1 % | 5,9 % |
| non_vegetal | 9,5 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | vegetal | non_vegetal |
|---|---|---|
| vegetal | 92,6 % | 7,6 % |
| non_vegetal | 7,4 % | 92,4 % |


### Niveau végétation, classes non comparables comptées comme végétales

**Version Vegestrate : natif**

| EVA \ Vegestrate (km²) | vegetal | non_vegetal |
|---|---|---|
| vegetal | 315,192 | 16,169 |
| non_vegetal | 18,764 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | vegetal | non_vegetal |
|---|---|---|
| vegetal | 95,1 % | 4,9 % |
| non_vegetal | 9,5 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | vegetal | non_vegetal |
|---|---|---|
| vegetal | 94,4 % | 8,3 % |
| non_vegetal | 5,6 % | 91,7 % |


### Niveau ligneux / herbacé

**Version Vegestrate : natif**

| EVA \ Vegestrate (km²) | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 124,282 | 22,150 | 8,158 |
| herbace | 12,944 | 75,223 | 6,464 |
| non_vegetal | 8,286 | 10,478 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 80,4 % | 14,3 % | 5,3 % |
| herbace | 13,7 % | 79,5 % | 6,8 % |
| non_vegetal | 4,2 % | 5,3 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 85,4 % | 20,5 % | 4,2 % |
| herbace | 8,9 % | 69,7 % | 3,4 % |
| non_vegetal | 5,7 % | 9,7 % | 92,4 % |

**Version Vegestrate : principal**

| EVA \ Vegestrate (km²) | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 120,612 | 25,820 | 8,158 |
| herbace | 10,379 | 77,787 | 6,464 |
| non_vegetal | 7,262 | 11,502 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 78,0 % | 16,7 % | 5,3 % |
| herbace | 11,0 % | 82,2 % | 6,8 % |
| non_vegetal | 3,7 % | 5,9 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 87,2 % | 22,4 % | 4,2 % |
| herbace | 7,5 % | 67,6 % | 3,4 % |
| non_vegetal | 5,3 % | 10,0 % | 92,4 % |

**Version Vegestrate : seuils_bas**

| EVA \ Vegestrate (km²) | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 120,612 | 25,820 | 8,158 |
| herbace | 10,379 | 77,787 | 6,464 |
| non_vegetal | 7,262 | 11,502 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 78,0 % | 16,7 % | 5,3 % |
| herbace | 11,0 % | 82,2 % | 6,8 % |
| non_vegetal | 3,7 % | 5,9 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 87,2 % | 22,4 % | 4,2 % |
| herbace | 7,5 % | 67,6 % | 3,4 % |
| non_vegetal | 5,3 % | 10,0 % | 92,4 % |

**Version Vegestrate : seuils_hauts**

| EVA \ Vegestrate (km²) | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 117,530 | 28,902 | 8,158 |
| herbace | 9,192 | 78,975 | 6,464 |
| non_vegetal | 6,575 | 12,189 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 76,0 % | 18,7 % | 5,3 % |
| herbace | 9,7 % | 83,5 % | 6,8 % |
| non_vegetal | 3,3 % | 6,2 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | ligneux | herbace | non_vegetal |
|---|---|---|---|
| ligneux | 88,2 % | 24,1 % | 4,2 % |
| herbace | 6,9 % | 65,8 % | 3,4 % |
| non_vegetal | 4,9 % | 10,2 % | 92,4 % |


### Niveau strates

**Version Vegestrate : natif**

| EVA \ Vegestrate (km²) | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 37,006 | 5,581 | 5,995 | 3,032 |
| arbuste | 6,402 | 14,681 | 12,722 | 4,244 |
| herbe | 5,909 | 7,035 | 75,223 | 6,464 |
| non_vegetal | 3,974 | 4,312 | 10,478 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 71,7 % | 10,8 % | 11,6 % | 5,9 % |
| arbuste | 16,8 % | 38,6 % | 33,4 % | 11,2 % |
| herbe | 6,2 % | 7,4 % | 79,5 % | 6,8 % |
| non_vegetal | 2,0 % | 2,2 % | 5,3 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 69,4 % | 17,7 % | 5,7 % | 1,6 % |
| arbuste | 12,0 % | 46,4 % | 12,2 % | 2,2 % |
| herbe | 11,1 % | 22,3 % | 72,0 % | 3,4 % |
| non_vegetal | 7,5 % | 13,6 % | 10,0 % | 92,8 % |

**Version Vegestrate : principal**

| EVA \ Vegestrate (km²) | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 39,094 | 3,002 | 6,486 | 3,032 |
| arbuste | 7,387 | 12,074 | 14,344 | 4,244 |
| herbe | 6,247 | 4,133 | 77,787 | 6,464 |
| non_vegetal | 4,454 | 2,808 | 11,502 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 75,7 % | 5,8 % | 12,6 % | 5,9 % |
| arbuste | 19,4 % | 31,7 % | 37,7 % | 11,2 % |
| herbe | 6,6 % | 4,4 % | 82,2 % | 6,8 % |
| non_vegetal | 2,3 % | 1,4 % | 5,9 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 68,4 % | 13,6 % | 5,9 % | 1,6 % |
| arbuste | 12,9 % | 54,8 % | 13,0 % | 2,2 % |
| herbe | 10,9 % | 18,8 % | 70,6 % | 3,4 % |
| non_vegetal | 7,8 % | 12,8 % | 10,4 % | 92,8 % |

**Version Vegestrate : seuils_bas**

| EVA \ Vegestrate (km²) | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 40,588 | 1,508 | 6,486 | 3,032 |
| arbuste | 12,022 | 7,440 | 14,344 | 4,244 |
| herbe | 7,439 | 2,940 | 77,787 | 6,464 |
| non_vegetal | 5,373 | 1,888 | 11,502 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 78,6 % | 2,9 % | 12,6 % | 5,9 % |
| arbuste | 31,6 % | 19,6 % | 37,7 % | 11,2 % |
| herbe | 7,9 % | 3,1 % | 82,2 % | 6,8 % |
| non_vegetal | 2,7 % | 1,0 % | 5,9 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 62,0 % | 10,9 % | 5,9 % | 1,6 % |
| arbuste | 18,4 % | 54,0 % | 13,0 % | 2,2 % |
| herbe | 11,4 % | 21,3 % | 70,6 % | 3,4 % |
| non_vegetal | 8,2 % | 13,7 % | 10,4 % | 92,8 % |

**Version Vegestrate : seuils_hauts**

| EVA \ Vegestrate (km²) | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 39,094 | 2,608 | 6,879 | 3,032 |
| arbuste | 7,387 | 10,178 | 16,241 | 4,244 |
| herbe | 6,247 | 2,945 | 78,975 | 6,464 |
| non_vegetal | 4,454 | 2,121 | 12,189 | 177,839 |

Part de chaque classe EVA (lignes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 75,7 % | 5,1 % | 13,3 % | 5,9 % |
| arbuste | 19,4 % | 26,7 % | 42,7 % | 11,2 % |
| herbe | 6,6 % | 3,1 % | 83,5 % | 6,8 % |
| non_vegetal | 2,3 % | 1,1 % | 6,2 % | 90,5 % |

Part de chaque classe Vegestrate (colonnes) :

| EVA \ Vegestrate | arbre | arbuste | herbe | non_vegetal |
|---|---|---|---|---|
| arbre | 68,4 % | 14,6 % | 6,0 % | 1,6 % |
| arbuste | 12,9 % | 57,0 % | 14,2 % | 2,2 % |
| herbe | 10,9 % | 16,5 % | 69,1 % | 3,4 % |
| non_vegetal | 7,8 % | 11,9 % | 10,7 % | 92,8 % |


### Conformité des polygones EVA à leur règle de couverture

La règle porte sur la part de la surface du polygone occupée par la strate cible selon Vegestrate (« ligneux » = arbre + arbuste).

| code_eva | version_vegestrate | regle | page_guide | polygones | part_polygones_conformes | part_surface_conforme | mediane |
|---|---|---|---|---|---|---|---|
| 111 | natif | arbre >= 75% | 3 | 60479 | 38,6 % | 52,6 % | 68,6 % |
| 112 | natif | arbuste >= 75% | 4 | 62629 | 5,3 % | 4,2 % | 36,5 % |
| 113 | natif | herbe >= 75% | 5 | 82253 | 45,2 % | 59,0 % | 72,4 % |
| 311 | natif | ligneux >= 75% | 22 | 5647 | 79,5 % | 98,3 % | 90,8 % |
| 312 | natif | ligneux >= 60% | 23 | 25 | 88,0 % | 86,4 % | 92,1 % |
| 321 | natif | ligneux >= 75% | 24 | 326 | 84,0 % | 92,7 % | 89,1 % |
| 341 | natif | ligneux >= 75% | 27 | 21 | 81,0 % | 69,0 % | 93,8 % |
| 351 | natif | ligneux >= 75% | 29 | 462 | 91,1 % | 98,6 % | 92,8 % |
| 361 | natif | ligneux < 25% | 30 | 34 | 35,3 % | 48,1 % | 31,2 % |
| 111 | principal | arbre >= 75% | 3 | 60479 | 48,7 % | 65,2 % | 74,3 % |
| 112 | principal | arbuste >= 75% | 4 | 62629 | 4,5 % | 3,7 % | 27,0 % |
| 113 | principal | herbe >= 75% | 5 | 82253 | 48,9 % | 64,4 % | 74,4 % |
| 311 | principal | ligneux >= 75% | 22 | 5647 | 74,2 % | 97,5 % | 89,0 % |
| 312 | principal | ligneux >= 60% | 23 | 25 | 84,0 % | 84,6 % | 89,6 % |
| 321 | principal | ligneux >= 75% | 24 | 326 | 81,6 % | 88,9 % | 88,8 % |
| 341 | principal | ligneux >= 75% | 27 | 21 | 61,9 % | 67,7 % | 93,8 % |
| 351 | principal | ligneux >= 75% | 29 | 462 | 83,5 % | 96,8 % | 91,0 % |
| 361 | principal | ligneux < 25% | 30 | 34 | 55,9 % | 64,7 % | 22,7 % |

### Accord selon la distance au bord des zones EVA

« dans EVA » : pixels à l'intérieur d'une zone EVA végétale (ou ligneuse), part que Vegestrate voit aussi végétale (ou ligneuse). « hors EVA » : pixels EVA non végétaux (ou non ligneux), part que Vegestrate voit végétale (ou ligneuse). La distance est mesurée au pixel le plus proche de l'autre côté du contour.

| niveau | cote | distance_au_bord | surface_km2 | part_vegestrate_positif |
|---|---|---|---|---|
| vegetation | hors EVA | 0-1 m | 21,298 | 35,6 % |
| vegetation | hors EVA | 1-2 m | 19,795 | 14,4 % |
| vegetation | hors EVA | 2-3 m | 17,595 | 7,8 % |
| vegetation | hors EVA | 3-5 m | 27,414 | 5,8 % |
| vegetation | hors EVA | 5-10 m | 37,725 | 6,2 % |
| vegetation | hors EVA | > 10 m | 72,775 | 4,1 % |
| vegetation | dans EVA | 0-1 m | 22,464 | 71,3 % |
| vegetation | dans EVA | 1-2 m | 20,412 | 87,6 % |
| vegetation | dans EVA | 2-3 m | 17,609 | 92,6 % |
| vegetation | dans EVA | 3-5 m | 27,428 | 95,0 % |
| vegetation | dans EVA | 5-10 m | 40,426 | 96,5 % |
| vegetation | dans EVA | > 10 m | 120,880 | 98,7 % |
| ligneux | hors EVA | 0-1 m | 19,980 | 25,5 % |
| ligneux | hors EVA | 1-2 m | 19,250 | 12,3 % |
| ligneux | hors EVA | 2-3 m | 17,849 | 7,3 % |
| ligneux | hors EVA | 3-5 m | 30,624 | 5,6 % |
| ligneux | hors EVA | 5-10 m | 53,097 | 5,5 % |
| ligneux | hors EVA | > 10 m | 150,434 | 2,8 % |
| ligneux | dans EVA | 0-1 m | 20,308 | 46,8 % |
| ligneux | dans EVA | 1-2 m | 17,959 | 62,6 % |
| ligneux | dans EVA | 2-3 m | 14,690 | 70,6 % |
| ligneux | dans EVA | 3-5 m | 21,016 | 76,7 % |
| ligneux | dans EVA | 5-10 m | 25,471 | 83,6 % |
| ligneux | dans EVA | > 10 m | 55,146 | 94,5 % |

### Végétation en petites taches (< 100 m²)

| niveau | groupe_eva | surface_petites_taches_km2 | surface_vegestrate_km2 | part_petites_taches |
|---|---|---|---|---|
| vegetation | total | 3,540 | 333,956 | 1,1 % |
| vegetation | absent | 0,000 | 0,000 | 0,0 % |
| vegetation | arbore | 0,221 | 48,581 | 0,5 % |
| vegetation | arbustif | 0,326 | 33,806 | 1,0 % |
| vegetation | boise_mixte | 0,006 | 64,045 | 0,0 % |
| vegetation | eau | 0,013 | 0,562 | 2,3 % |
| vegetation | herbace | 0,372 | 88,167 | 0,4 % |
| vegetation | non_comparable | 0,020 | 80,593 | 0,0 % |
| vegetation | non_vegetal | 2,583 | 18,201 | 14,2 % |
| ligneux | total | 8,913 | 140,098 | 6,4 % |
| ligneux | absent | 0,000 | 0,000 | 0,0 % |
| ligneux | arbore | 0,941 | 42,096 | 2,2 % |
| ligneux | arbustif | 2,374 | 19,461 | 12,2 % |
| ligneux | boise_mixte | 0,164 | 59,055 | 0,3 % |
| ligneux | eau | 0,008 | 0,283 | 2,8 % |
| ligneux | herbace | 2,581 | 10,379 | 24,9 % |
| ligneux | non_comparable | 0,225 | 1,845 | 12,2 % |
| ligneux | non_vegetal | 2,620 | 6,979 | 37,5 % |

### Couverture par commune

Parts calculées sur la surface comparable de chaque commune (classes EVA non comparables et zone exclue retirées pour les deux sources). Le détail par IRIS est dans `couverture_iris.csv`.

| commune | surface_comparable_vegetal_km2 | part_vegetal_eva | part_vegetal_vegestrate | part_ligneux_eva | part_ligneux_vegestrate |
|---|---|---|---|---|---|
| Saint-Priest | 23,926 | 45,2 % | 46,6 % | 21,4 % | 20,1 % |
| Meyzieu | 19,330 | 53,9 % | 53,5 % | 31,8 % | 27,5 % |
| Vaulx-en-Velin | 18,930 | 48,5 % | 50,2 % | 31,9 % | 30,1 % |
| Quincieux | 8,104 | 62,2 % | 63,3 % | 22,6 % | 20,5 % |
| Givors | 14,813 | 71,7 % | 72,3 % | 45,7 % | 42,8 % |
| Décines-Charpieu | 13,494 | 45,9 % | 47,4 % | 24,0 % | 21,8 % |
| Vénissieux | 14,025 | 36,4 % | 37,6 % | 20,0 % | 16,5 % |
| Villeurbanne | 14,788 | 30,9 % | 33,1 % | 20,7 % | 18,3 % |
| Rillieux-la-Pape | 11,570 | 59,2 % | 60,0 % | 42,1 % | 37,5 % |
| Dardilly | 12,958 | 75,8 % | 76,2 % | 43,0 % | 39,9 % |
| Saint-Genis-Laval | 11,965 | 63,4 % | 65,3 % | 34,5 % | 30,3 % |
| Jonage | 7,692 | 67,8 % | 67,1 % | 41,0 % | 34,7 % |
| Corbas | 8,297 | 43,5 % | 43,5 % | 17,3 % | 13,4 % |
| Mions | 8,339 | 51,6 % | 52,4 % | 22,0 % | 18,8 % |
| Chassieu | 8,855 | 46,3 % | 46,8 % | 18,1 % | 16,3 % |
| Feyzin | 7,688 | 51,1 % | 50,6 % | 31,6 % | 26,5 % |
| Caluire-et-Cuire | 9,815 | 52,8 % | 53,0 % | 41,4 % | 34,7 % |
| Bron | 10,066 | 47,9 % | 48,9 % | 29,1 % | 24,6 % |
| Lyon 7e Arrondissement | 9,451 | 22,8 % | 24,7 % | 13,1 % | 12,2 % |
| Irigny | 7,117 | 66,8 % | 66,7 % | 40,9 % | 36,0 % |
| Genay | 5,391 | 59,5 % | 60,1 % | 31,0 % | 27,5 % |
| Écully | 8,414 | 65,6 % | 66,7 % | 45,0 % | 42,4 % |
| Cailloux-sur-Fontaines | 3,202 | 69,8 % | 71,1 % | 34,9 % | 32,7 % |
| La Tour-de-Salvagny | 7,699 | 80,8 % | 81,2 % | 47,5 % | 43,6 % |
| Francheville | 8,118 | 74,0 % | 75,2 % | 54,4 % | 49,8 % |
| Solaize | 6,041 | 56,0 % | 55,5 % | 36,7 % | 33,0 % |
| Tassin-la-Demi-Lune | 7,334 | 66,1 % | 67,9 % | 49,1 % | 44,4 % |
| Limonest | 7,020 | 75,4 % | 74,7 % | 46,3 % | 38,2 % |
| Saint-Didier-au-Mont-d'Or | 7,491 | 76,3 % | 76,8 % | 49,9 % | 43,9 % |
| Lyon 9e Arrondissement | 7,436 | 42,6 % | 44,4 % | 33,6 % | 30,7 % |
| Montanay | 3,196 | 70,6 % | 70,8 % | 38,1 % | 33,8 % |
| Saint-Cyr-au-Mont-d'Or | 6,706 | 74,2 % | 76,2 % | 53,0 % | 48,3 % |
| Sainte-Foy-lès-Lyon | 6,831 | 63,1 % | 64,2 % | 46,3 % | 40,0 % |
| Lyon 8e Arrondissement | 6,722 | 30,4 % | 33,6 % | 19,5 % | 17,6 % |
| Lyon 3e Arrondissement | 6,347 | 26,3 % | 29,1 % | 20,5 % | 18,3 % |
| Lyon 5e Arrondissement | 6,131 | 50,3 % | 51,4 % | 39,6 % | 33,8 % |
| Saint-Fons | 5,955 | 33,5 % | 34,5 % | 17,4 % | 14,6 % |
| Grigny | 5,473 | 57,1 % | 57,5 % | 35,5 % | 32,5 % |
| Lissieu | 5,316 | 77,1 % | 78,3 % | 39,6 % | 36,4 % |
| Neuville-sur-Saône | 4,684 | 60,5 % | 61,8 % | 41,5 % | 38,1 % |
| Saint-Germain-au-Mont-d'Or | 4,103 | 71,7 % | 72,6 % | 46,3 % | 43,6 % |
| Marcy-l'Étoile | 5,224 | 72,3 % | 73,3 % | 38,7 % | 34,1 % |
| Charly | 4,178 | 72,4 % | 71,5 % | 32,4 % | 26,3 % |
| Sathonay-Village | 2,201 | 75,3 % | 74,8 % | 47,7 % | 41,9 % |
| Craponne | 4,690 | 58,2 % | 59,9 % | 38,2 % | 32,9 % |
| Oullins-Pierre-Bénite | 4,408 | 46,5 % | 49,1 % | 35,2 % | 30,2 % |
| Vernaison | 3,562 | 70,4 % | 70,3 % | 49,8 % | 45,7 % |
| Charbonnières-les-Bains | 4,027 | 70,1 % | 71,2 % | 49,5 % | 46,2 % |
| Saint-Genis-les-Ollières | 3,684 | 71,4 % | 73,2 % | 37,1 % | 33,3 % |
| Collonges-au-Mont-d'Or | 3,684 | 58,3 % | 60,4 % | 43,0 % | 37,6 % |
| Lyon 6e Arrondissement | 3,754 | 34,5 % | 35,4 % | 27,0 % | 24,9 % |
| Lyon 2e Arrondissement | 3,444 | 10,3 % | 12,5 % | 7,6 % | 7,4 % |
| Couzon-au-Mont-d'Or | 3,048 | 77,2 % | 77,5 % | 63,3 % | 59,4 % |
| Curis-au-Mont-d'Or | 2,747 | 86,9 % | 86,5 % | 62,3 % | 58,8 % |
| Fleurieu-sur-Saône | 2,389 | 74,3 % | 73,9 % | 57,7 % | 54,4 % |
| Lyon 4e Arrondissement | 2,924 | 37,8 % | 39,6 % | 31,7 % | 28,3 % |
| Poleymieux-au-Mont-d'Or | 2,319 | 90,0 % | 87,4 % | 59,1 % | 44,9 % |
| Fontaines-Saint-Martin | 2,328 | 75,4 % | 74,6 % | 48,2 % | 43,3 % |
| Albigny-sur-Saône | 2,606 | 62,5 % | 63,1 % | 47,0 % | 42,6 % |
| Saint-Romain-au-Mont-d'Or | 2,532 | 86,1 % | 86,4 % | 71,6 % | 68,9 % |
| Champagne-au-Mont-d'Or | 2,564 | 55,4 % | 56,8 % | 40,2 % | 36,2 % |
| Fontaines-sur-Saône | 2,292 | 61,9 % | 61,6 % | 47,0 % | 41,3 % |
| Sathonay-Camp | 1,720 | 60,1 % | 61,0 % | 44,4 % | 40,0 % |
| La Mulatière | 1,836 | 40,9 % | 42,7 % | 31,7 % | 28,3 % |
| Lyon 1er Arrondissement | 1,533 | 21,4 % | 22,2 % | 19,0 % | 17,2 % |
| Rochetaillée-sur-Saône | 1,299 | 59,8 % | 60,6 % | 41,5 % | 38,4 % |

### Accord par zone climatique locale (LCZ)

| niveau | lcz | surface_km2 | accord | desaccord_quantite | desaccord_allocation |
|---|---|---|---|---|---|
| vegetation | Sans LCZ | 0,917 | 94,3 % | 0,7 % | 5,0 % |
| vegetation | 1 - Ensemble compact de tours | 0,222 | 93,5 % | 1,8 % | 4,7 % |
| vegetation | 2 - Ensemble compact d'immeubles | 8,951 | 93,2 % | 2,1 % | 4,8 % |
| vegetation | 3 - Ensemble compact de maisons | 1,439 | 91,7 % | 3,4 % | 4,9 % |
| vegetation | 4 - Ensemble de tours espacées | 0,510 | 88,8 % | 2,0 % | 9,2 % |
| vegetation | 5 - Ensemble d'immeubles espacés | 28,755 | 89,3 % | 2,2 % | 8,6 % |
| vegetation | 6 - Ensemble de maisons espacées | 25,054 | 88,2 % | 2,0 % | 9,8 % |
| vegetation | 8 - Bâtiments bas de grande emprise | 51,952 | 92,8 % | 1,9 % | 5,3 % |
| vegetation | 9 - Implantation diffuse de maisons | 84,816 | 88,5 % | 0,3 % | 11,2 % |
| vegetation | A - Espace densément arboré | 88,597 | 96,6 % | 0,3 % | 3,2 % |
| vegetation | B - Espace arboré clairsemé | 31,980 | 94,0 % | 0,5 % | 5,5 % |
| vegetation | C - Espace végétalisé hétérogène | 23,941 | 96,0 % | 0,3 % | 3,7 % |
| vegetation | D - Végétation basse | 14,601 | 95,6 % | 1,4 % | 3,0 % |
| vegetation | E - Sol imperméable naturel ou artificiel | 59,661 | 90,7 % | 1,6 % | 7,7 % |
| vegetation | F - Sol nu perméable | 5,652 | 94,0 % | 2,7 % | 3,4 % |
| vegetation | G - Surface en eau | 18,776 | 97,2 % | 1,2 % | 1,7 % |
| ligneux_herbace | Sans LCZ | 0,917 | 85,7 % | 5,0 % | 9,4 % |
| ligneux_herbace | 1 - Ensemble compact de tours | 0,222 | 91,5 % | 2,7 % | 5,8 % |
| ligneux_herbace | 2 - Ensemble compact d'immeubles | 8,951 | 90,0 % | 3,5 % | 6,5 % |
| ligneux_herbace | 3 - Ensemble compact de maisons | 1,439 | 87,8 % | 4,7 % | 7,4 % |
| ligneux_herbace | 4 - Ensemble de tours espacées | 0,510 | 82,3 % | 3,2 % | 14,5 % |
| ligneux_herbace | 5 - Ensemble d'immeubles espacés | 28,755 | 81,4 % | 5,3 % | 13,3 % |
| ligneux_herbace | 6 - Ensemble de maisons espacées | 25,054 | 78,0 % | 8,1 % | 13,9 % |
| ligneux_herbace | 8 - Bâtiments bas de grande emprise | 51,952 | 89,4 % | 1,9 % | 8,8 % |
| ligneux_herbace | 9 - Implantation diffuse de maisons | 84,816 | 74,2 % | 8,9 % | 16,9 % |
| ligneux_herbace | A - Espace densément arboré | 88,597 | 88,7 % | 3,8 % | 7,5 % |
| ligneux_herbace | B - Espace arboré clairsemé | 31,980 | 81,5 % | 4,2 % | 14,2 % |
| ligneux_herbace | C - Espace végétalisé hétérogène | 23,941 | 87,4 % | 3,2 % | 9,4 % |
| ligneux_herbace | D - Végétation basse | 14,601 | 90,2 % | 3,5 % | 6,3 % |
| ligneux_herbace | E - Sol imperméable naturel ou artificiel | 59,661 | 86,3 % | 2,9 % | 10,8 % |
| ligneux_herbace | F - Sol nu perméable | 5,652 | 86,6 % | 5,3 % | 8,1 % |
| ligneux_herbace | G - Surface en eau | 18,776 | 96,4 % | 1,2 % | 2,4 % |

### Lieux d'exemple

Pour chaque lieu (1 km²), les rasters `exemples/<lieu>_*.tif` et leurs styles QGIS `.qml` : groupes EVA, Vegestrate natif, Vegestrate recoupé, accord végétation et accord par strate. Répartition des catégories d'accord par strate :

| categorie | Campagne, Monts d'Or (hors camp militaire) | Centre-ville, Presqu'île (Lyon 2e) | Grand parc urbain, Parc de la Tête d'Or | Industrie, Vallée de la chimie (Saint-Fons) | Périurbain pavillonnaire, Tassin-la-Demi-Lune |
|---|---|---|---|---|---|
| EVA boisé multistrate, Vegestrate herbe | 1,2 % | 0,0 % | 0,0 % | 0,0 % | 0,1 % |
| EVA boisé multistrate, Vegestrate ligneux | 37,9 % | 0,0 % | 0,3 % | 0,0 % | 1,7 % |
| EVA non comparable | 3,5 % | 0,0 % | 0,0 % | 0,0 % | 0,0 % |
| Même strate | 28,0 % | 7,8 % | 47,7 % | 17,5 % | 39,4 % |
| Non végétal pour les deux | 13,9 % | 87,0 % | 38,3 % | 70,7 % | 32,7 % |
| Végétal pour EVA seul | 2,3 % | 1,8 % | 2,8 % | 2,8 % | 4,2 % |
| Végétal pour Vegestrate seul | 2,7 % | 2,0 % | 3,9 % | 5,0 % | 6,5 % |
| Végétal pour les deux, strate différente | 10,5 % | 1,4 % | 6,9 % | 3,9 % | 15,3 % |

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

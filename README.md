# Inventaire stratifié du végétal

Vous trouverez dans ce repos un wrapper pour [FLAIR-HUB](https://github.com/IGNF/FLAIR-HUB) pour faire de l'inférence directement et des fonctions pour faire du fine-tunning. Les poids de FLAIR-HUB sont sur [HuggingFace](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGB_swinlarge-upernet).

## Origine des données

Les données d'inventaire du végétal stratifié ont été produites par un travail conjoint entre [TelesCoop](https://www.telescoop.fr/) et le [LIRIS](https://liris.cnrs.fr/). Le LIRIS a produit une note sur les méthodes permettant de produire un inventaire de végétation en contexte urbain qui se trouve [ici](https://github.com/VCityTeam/UD-IArbre-Research/blob/master/vegetalisation/Pr%C3%A9sentation%20Cotech%2020-11-2025%20Segmentation%20V%C3%A9g%C3%A9talisation.pdf). Puis, chez TelesCoop nous avons industrialisé la démarche et proposé un pipeline automatisé.


### Données d'entrées

Les données d'entrée proviennent de data.grandlyon :

- Les nuages de point [LIDAR de 2023](https://data.grandlyon.com/portail/fr/jeux-de-donnees/nuage-de-points-lidar-2023-de-la-metropole-de-lyon/info);
- Les [orthophotos 2023](https://data.grandlyon.com/portail/fr/jeux-de-donnees/orthophotographie-2023-de-la-metropole-de-lyon/info).

Il y a aussi des scripts pour faire tourner sur les ortho 2023 de l'IGN et le le dernier LIDAR disponible qui est 2021 (`src/data_preparation/prepare_training_data_IGN.py`)
## Méthode

Nous utilisons d'un côté la classification des nuages de points LIDAR  (`src/data_preparation/prepare_training_data_grandlyon.py`) et par ailleurs la classification des orthophotos à l'aide de [FLAIR-HUB](https://github.com/IGNF/FLAIR-HUB) de l'IGN (`src/inference/inference_flair_context.py`) puis les 2 classifications sont fusionnées (`src/postprocessing/merge_classifications.py`).
La précision de la classification, taille d'un pixel, est un carré de **80cmsx80cms**. Cette résolution a été choisie car le modèle FLAIR-HUB a été entraîné sur des images à cette résolution.

### Classification des nuages de points LIDAR
> `src/data_preparation/prepare_training_data_grandlyon.py`
Les nuages de points sont déjà classées, nous récupérons donc les points correspondants aux catégories 4 `végétation moyenne de 1,5-5 m`, 5 `végétation haute 5-15 m` et 8 `végétation haute > 15 m`. 5 et 8 sont rassemblées pour définir une seule catégorie végétation haute. Le reste est dans la catégorie `Autre`.
La classification des végétation basses ne fonctionne pas bien avec le LIDAR, nous ne l'utilisons pas.
Le nuage de point est rasterisé en utilisant une résolution de 0.8m.

### Classification des orthophotos avec FLAIR-HUB
> `src/inference/inference_flair_context.py`
Nous utilisons la version avec encoder Swin large, decoder UPerNet et en RGB. Les poids sont disponibles sur [HuggingFace)(https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGB_swinlarge-upernet).
La résolution des orthophotos est réduite à l'aide d'une interpolation bi-cubique pour passer d'une résolution de 5cms à 80cms. Ce choix s'eplique de deux façons :

- Garder une résolution existante dans les années précédentes afin de pouvoir avoir des analyses diachroniques;
- Avoir la même résolution que les données d'entraînement de FLAIR-HUB afin de maximiser les performances.

Les orhtophotos sont découpées en patch de 384 pixels et le recouvrement entre patchs est de 256 pixels. Les patchs utilisent les données de plusieurs dalles afin d'éviter les effets de bord. Nous utilisons une test-time augmentation (TTA) avec des flips horizontaux et verticaux pour plus de robustesse. Parmis les 20 classes, nous ne conservons que celles relatives à la végétation haute, moyenne et basse.

### Fusion des résultats
> `src/postprocessing/merge_classifications.py`
Nous partons des résultats LIDAR pour la végétation moyenne et haute, auquelle on ajoute le résultat de végétation basse de FLAIR-HUB. Finalement on met à jour les zones classées comme `Autre` par le LIDAR mais qui sont de la végétation moyenne et haute pour FLAIR-HUB. Ces zones correspondent souvent à des zones proches des bâtiments qui sont mal détectées par le LIDAR et mieux avec les orhtophotos.

### Vectorisation du résultat
Le format de sortie est TIF qui peut être vectorisé facilement avec [GDAL](https://gdal.org/en/stable/) et son [API Python](https://gdal.org/en/stable/api/python/index.html).
Pour plus de détails, voir [ici](https://github.com/TelesCoop/vegestrate/blob/main/pipeline_grandlyon.py#L26).

## Limites

La qualité du résultat est très dépendante du LIDAR qui reste la meilleure manière de classifier la végétation, hors zones herbacées, de manière précise (résolution de l'ordre du mètre).
La métropole de Lyon produit une couverture du territoire en THD (100 points par m2 en zone urbaine dense et 30 ailleurs) ce qui permet une classification très précise. En zone urbaine dense, c'est parfois trop car on a des points qui traversent le couvert arboré et se retrouvent classés en zone herbacée qui est en dessous.
Comme évoqué plus haut, le LIDAR pert en précision dans les zones proches des bâtiments.

Le modèle FLAIR-HUB permet à une résolution très compétitive, 80cm, des détections de zones herbacés très précises. Le modèle se comporte également très bien dans les zones proches des bâtiments où le LIDAR est moins bon.

Nous ne disposons pas de vérité terrain à l'échelle de la Métropole, car cette donnée n'existe pas, qui permetterait de calculer des métriques quantitatives de performance. Pour évaluer la performance nous sommes dépendants d'évaluations qualitatives avec les orthophotos en dessous de plan ou à l'aide d'experts d'un territoire précis.


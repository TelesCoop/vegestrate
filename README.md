# 🌳 Inventaire stratifié du végétal

Vous trouverez dans ce repos un wrapper pour [FLAIR-HUB](https://github.com/IGNF/FLAIR-HUB) pour faire de l'inférence directement et des fonctions pour faire du fine-tunning. Les poids de FLAIR-HUB sont sur [HuggingFace](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGB_swinlarge-upernet).
Si vous réutilisez les poids ou le modèle FLAIR-HUB, un like sur [GitHub](https://github.com/IGNF/FLAIR-HUB)/[Hugging Face](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGB_swinlarge-upernet) leur permet de valoriser leurs travaux, en interne comme en externe !

Ce travail a été réalisé dans le cadre du projet [IA.rbre](https://iarbre.fr) et vous pouvez trouver les autres calque sur notre plateforme [carte.iarbre.fr](https://carte.iarbre.fr).

## 📑 Table des matières

- [📊 Origine des données](#-origine-des-données)
- [⚙️ Méthode](#️-méthode)
- [⚠️ Limites](#️-limites)
- [📦 Installation](#-installation)
- [🚀 Pipeline GrandLyon](#-pipeline-grandlyon)
- [🛠️ Configuration de Pre-Commit](#️-configuration-de-pre-commit)
- [🤝 Contribution](#-contribution)
- [📜 Datapaper](#datapaper-flairhub)

---

## 📊 Origine des données

Les données d'inventaire du végétal stratifié ont été produites par un travail conjoint entre [TelesCoop](https://www.telescoop.fr/) et le [LIRIS](https://liris.cnrs.fr/). Le LIRIS a produit une note sur les méthodes permettant de produire un inventaire de végétation en contexte urbain qui se trouve [ici](https://github.com/VCityTeam/UD-IArbre-Research/blob/master/vegetalisation/Pr%C3%A9sentation%20Cotech%2020-11-2025%20Segmentation%20V%C3%A9g%C3%A9talisation.pdf). Puis, chez TelesCoop nous avons industrialisé la démarche et proposé un pipeline automatisé.


### 📥 Données d'entrées

Les données d'entrée proviennent de data.grandlyon :

- 📡 Les nuages de point [LIDAR de 2023](https://data.grandlyon.com/portail/fr/jeux-de-donnees/nuage-de-points-lidar-2023-de-la-metropole-de-lyon/info);
- 📸 Les [orthophotos 2023](https://data.grandlyon.com/portail/fr/jeux-de-donnees/orthophotographie-2023-de-la-metropole-de-lyon/info).

Il y a aussi des scripts pour faire tourner sur les ortho 2023 de l'IGN et le le dernier LIDAR disponible qui est 2021 (`src/data_preparation/prepare_training_data_IGN.py`)
## ⚙️ Méthode

Nous utilisons d'un côté la classification des nuages de points LIDAR  (`src/data_preparation/prepare_training_data_grandlyon.py`) et par ailleurs la classification des orthophotos à l'aide de [FLAIR-HUB](https://github.com/IGNF/FLAIR-HUB) de l'IGN (`src/inference/inference_flair_context.py`) puis les 2 classifications sont fusionnées (`src/postprocessing/merge_classifications.py`).

La précision de la classification, taille d'un pixel, est un carré de **80cmsx80cms**. Cette résolution a été choisie car le modèle FLAIR-HUB a été entraîné sur des images à cette résolution.

### 📡 Classification des nuages de points LIDAR
> `src/data_preparation/prepare_training_data_grandlyon.py`

Les nuages de points sont déjà classées, nous récupérons donc les points correspondants aux catégories 4 `végétation moyenne de 1,5-5 m`, 5 `végétation haute 5-15 m` et 8 `végétation haute > 15 m`. 5 et 8 sont rassemblées pour définir une seule catégorie végétation haute. Le reste est dans la catégorie `Autre`.

La classification des végétation basses ne fonctionne pas bien avec le LIDAR, nous ne l'utilisons pas.
Le nuage de point est rasterisé en utilisant une résolution de 0.8m.

### 🤖 Classification des orthophotos avec FLAIR-HUB
> `src/inference/inference_flair_context.py`

Nous utilisons la version avec encoder Swin large, decoder UPerNet et en RGB. Les poids sont disponibles sur [HuggingFace)(https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGB_swinlarge-upernet).

La résolution des orthophotos est réduite à l'aide d'une interpolation bi-cubique pour passer d'une résolution de 5cms à 80cms. Ce choix s'eplique de deux façons :

- 📅 Garder une résolution existante dans les années précédentes afin de pouvoir avoir des analyses diachroniques;
- 🎯 Avoir la même résolution que les données d'entraînement de FLAIR-HUB afin de maximiser les performances.

Les orhtophotos sont découpées en patch de 384 pixels et le recouvrement entre patchs est de 256 pixels. Les patchs utilisent les données de plusieurs dalles afin d'éviter les effets de bord. Nous utilisons une test-time augmentation (TTA) avec des flips horizontaux et verticaux pour plus de robustesse. Parmis les 20 classes, nous ne conservons que celles relatives à la végétation haute, moyenne et basse.

### 🔀 Fusion des résultats
> `src/postprocessing/merge_classifications.py`

Nous partons des résultats LIDAR pour la végétation moyenne et haute, auquelle on ajoute le résultat de végétation basse de FLAIR-HUB. Finalement on met à jour les zones classées comme `Autre` par le LIDAR mais qui sont de la végétation moyenne et haute pour FLAIR-HUB.
Ces zones correspondent souvent à des zones proches des bâtiments qui sont mal détectées par le LIDAR et mieux avec les orhtophotos.

### 🗺️ Vectorisation du résultat
Le format de sortie est TIF qui peut être vectorisé facilement avec [GDAL](https://gdal.org/en/stable/) et son [API Python](https://gdal.org/en/stable/api/python/index.html).
Pour plus de détails, voir [ici](https://github.com/TelesCoop/vegestrate/blob/main/pipeline_grandlyon.py#L26).

## ⚠️ Limites

La qualité du résultat est très dépendante du LIDAR qui reste la meilleure manière de classifier la végétation, hors zones herbacées, de manière précise (résolution de l'ordre du mètre).

La métropole de Lyon produit une couverture du territoire en THD (100 points par m2 en zone urbaine dense et 30 ailleurs) ce qui permet une classification très précise. En zone urbaine dense, c'est parfois trop car on a des points qui traversent le couvert arboré et se retrouvent classés en zone herbacée qui est en dessous.
Comme évoqué plus haut, le LIDAR pert en précision dans les zones proches des bâtiments.

Le modèle FLAIR-HUB permet à une résolution très compétitive, 80cm, des détections de zones herbacés très précises. Le modèle se comporte également très bien dans les zones proches des bâtiments où le LIDAR est moins bon.

Nous ne disposons pas de vérité terrain à l'échelle de la Métropole, car cette donnée n'existe pas, qui permetterait de calculer des métriques quantitatives de performance. Pour évaluer la performance nous sommes dépendants d'évaluations qualitatives avec les orthophotos en dessous de plan ou à l'aide d'experts d'un territoire précis.

## 📦 Installation

### 1. Installer le package

Depuis la racine du projet, installer le package en mode éditable :

```bash
pip install -e .
```

Cette commande installe le package `vegestrate` et configure correctement tous les imports `src.*` utilisés dans le projet.

### 2. Dépendances système

**GDAL** (requis pour la vectorisation uniquement) :

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev
sudo apt install libpython3.XX-dev python3.XX-dev

# Puis installer les bindings Python avec la version correspondante
pip install gdal==$(gdal-config --version)
```

Si vous n'utilisez pas la fonctionnalité de vectorisation, GDAL n'est pas nécessaire.

### 3. Dépendances Python

Installer toutes les dépendances Python listées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

**Note importante :** FLAIR-HUB sera installé depuis le dépôt GitHub :

### Structure du package

Après installation, la structure du package est la suivante :

```
vegestrate/
├── src/                    # Package principal (installé en tant que 'src')
│   ├── core/              # Utilitaires LiDAR et raster
│   ├── flairhub_utils/    # Utilitaires pour le modèle FLAIR-HUB
│   ├── inference/         # Modules d'inférence
│   ├── data_preparation/  # Scripts de préparation des données
│   └── postprocessing/    # Outils de post-traitement
├── pyproject.toml         # Configuration du package
├── setup.py               # Script de setup
└── requirements.txt       # Dépendances
```

### Commandes disponibles

Après installation, plusieurs commandes sont disponibles :

```bash
python src/data_preparation/update_manifest_grandlyon
python src/postprocessing/vectorize_raster -i input.tif -o output.gpkg
python pipeline_grandlyon.py --help
```

### Patterns d'imports

Tous les modules utilisent des imports absolus depuis `src` :

```python
from src.core import create_classification_map
from src.flairhub_utils import load_flair_model
```

## 🚀 Pipeline GrandLyon

Le pipeline complet est orchestré par `pipeline_grandlyon.py` et piloté par un fichier de configuration YAML. Un fichier d'état JSON enregistre le résultat de chaque phase et permet la reprise automatique après interruption.

### Utilisation

```bash
# Premier lancement (utilise pipeline_config.yaml par défaut)
python pipeline_grandlyon.py
# ou
vegestrate-pipeline

# Fichier de configuration personnalisé → état dans mon_run_state.json
python pipeline_grandlyon.py --config mon_run.yaml

# Forcer la ré-exécution d'une ou plusieurs phases spécifiques
python pipeline_grandlyon.py --force flair_inference
python pipeline_grandlyon.py --force flair_inference lidar_flair_merge

# Relancer toutes les phases depuis zéro
python pipeline_grandlyon.py --force all
```

### Configuration (`pipeline_config.yaml`)

Tous les paramètres sont regroupés dans un seul fichier YAML, versionnable et reproductible :

```yaml
pipeline:
  output_name: lyon          # Préfixe des sorties (répertoires, fichiers TIF/SHP)
  splits: [test]             # Splits à traiter : train, test, ou les deux

data:                        # prepare_training_data_grandlyon.py
  manifest: data/dataset_manifest_grandlyon.json
  resolution: 0.2            # Résolution LiDAR rasterisée (mètres)
  workers: 14                # Workers parallèles
  ir_mosaic: null            # Chemin vers la mosaïque IR (optionnel)
  download_ir: false         # Télécharger la mosaïque IR automatiquement

inference:                   # inference_flair_context.py
  checkpoint: FLAIR-HUB_LC-A_RGB_swinlarge-upernet.safetensors
  download_checkpoint: false # Télécharger depuis HuggingFace si absent
  tile_size: 512             # Taille des tuiles de traitement (pixels)
  overlap: 256               # Chevauchement entre tuiles (50 % → blending lisse)
  grid_step: 5               # Pas de grille entre dalles voisines
  tta: true                  # Test-Time Augmentation (×4 temps, meilleure précision)
  tta_modes: null            # null = [hflip, vflip, hvflip]
  batch_size: 8              # Tuiles traitées en parallèle sur GPU
  fp16: true                 # Précision mixte FP16
  compile: true              # Optimisation torch.compile
  use_ir: false              # Utiliser le canal infrarouge ([IR,R,G] au lieu de RGB)
  class_bias: null           # Biais par classe, ex: ["1:2.0"] pour booster les herbacées
  herb_margin: 3.0           # Récupération herbacée : marge sur le logit "else"

merge:                       # merge_tifs.py
  strategy: mode             # Stratégie de fusion des dalles : mode | last
  smooth: false              # Lissage des artefacts de jointure entre dalles
  smooth_iterations: 3       # Itérations de lissage
  smooth_cores: 3            # Cœurs CPU alloués au lissage
  resample_mismatch: false   # Rééchantillonner automatiquement les dalles hétérogènes

vectorization:               # vectorize_raster.py
  format: ESRI Shapefile     # Format vecteur : ESRI Shapefile | GPKG | GeoJSON
  eight_connected: false     # Connexité 8 pour la polygonisation (diagonales incluses)
  field_name: class          # Nom du champ attributaire dans le vecteur de sortie

phases:                      # Activer / désactiver chaque phase
  data_preparation: true
  flair_inference: true
  lidar_flair_merge: true
  final_merge: true
  vectorization: false       # Désactivée par défaut (requiert GDAL)
```

### Phases et ordre d'exécution

| # | Nom | Bloquante | Description |
|---|-----|-----------|-------------|
| 1 | `data_preparation` | ✓ | Rasterisation LiDAR + extraction orthophotos depuis le manifeste |
| 2 | `flair_inference` | ✓ | Inférence FLAIR-HUB avec contexte des dalles voisines |
| 3 | `lidar_flair_merge` | ✓ | Fusion pixel-à-pixel des classifications LiDAR et FLAIR |
| 4 | `final_merge` | ✓ | Assemblage de toutes les dalles en un raster unique |
| 5 | `vectorization` | — | Polygonisation du raster final (optionnelle, non bloquante) |

Une phase **bloquante** qui échoue arrête le pipeline. La vectorisation échoue avec un avertissement sans bloquer.

### Suivi d'état et reprise automatique

Le fichier `{config_stem}_state.json` (ex : `pipeline_config_state.json`) est créé automatiquement et mis à jour après chaque phase :

```json
{
  "config_hash": "a3f8bc12",
  "phases": {
    "data_preparation": {
      "status": "success",
      "start_time": "2026-03-03T14:30:00+00:00",
      "end_time": "2026-03-03T14:45:01+00:00",
      "duration_seconds": 879.0
    },
    "flair_inference": {
      "status": "failed",
      "error": "Checkpoint not found"
    }
  }
}
```

Les statuts possibles sont : `pending` · `running` · `success` · `failed` · `skipped`.

Comportement à la reprise :
- Les phases `success` sont sautées automatiquement
- Une phase `running` au démarrage (pipeline interrompu) est relancée
- Un changement de config depuis le dernier lancement affiche : `⚠ Config changed since last run — use --force all to rerun everything`

---

## 🛠️ Configuration de Pre-Commit

1. **Installer pre-commit** :

```bash
pip install pre-commit
```

2. **Installer les hooks** :

```bash
pre-commit install
```

3. **Exécuter manuellement les hooks (optionnel)** :

```bash
pre-commit run --all-files
```

C'est tout ! Maintenant, à chaque commit, `pre-commit` vérifiera automatiquement votre code. 🧹✨

## 🤝 Contribution

Si vous avez des idées, des bugs ou des demandes de fonctionnalités, n'hésitez pas à ouvrir une [issue](https://github.com/TelesCoop/iarbre/issues).

Vous pouvez également contribuer directement en proposant de nouvelles fonctionnalités :

1. **Forker le dépôt**
2. **Créer une branche de fonctionnalité** : `git checkout -b ma-fonctionnalite-geniale`
3. **Valider vos modifications** : `git commit -m "Ajouter une fonctionnalité géniale"`
4. **Pousser votre branche** : `git push origin ma-fonctionnalite-geniale`
5. **Ouvrir une Pull Request**

## Datapaper FlairHub

Nous n'avons pas utilisé le dataset FLAIR-HUB mais le modèle que nous utilisons a été entraîné dessus.

```bibtex
@article{ign2025flairhub,
  doi = {10.48550/arXiv.2506.07080},
  url = {https://arxiv.org/abs/2506.07080},
  author = {Garioud, Anatol and Giordano, Sébastien and David, Nicolas and Gonthier, Nicolas},
  title = {FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping},
  publisher = {arXiv},
  year = {2025}
}
```

```
Anatol Garioud, Sébastien Giordano, Nicolas David, Nicolas Gonthier.
FLAIR-HUB: Large-scale Multimodal Dataset for Land Cover and Crop Mapping. (2025).
DOI: https://doi.org/10.48550/arXiv.2506.07080
```

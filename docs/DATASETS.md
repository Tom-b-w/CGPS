# Dataset layout

Datasets are not redistributed. Set `--data-root` to a directory with the
layout below. Split JSON files follow the public CoOp/TDA convention.

```text
data/
├── caltech-101/
│   ├── 101_ObjectCategories/
│   └── split_zhou_Caltech101.json
├── dtd/
│   ├── images/
│   └── split_zhou_DescribableTextures.json
├── eurosat/
│   ├── 2750/
│   └── split_zhou_EuroSAT.json
├── fgvc_aircraft/
│   ├── images/
│   └── images_variant_test.txt
├── food-101/
│   ├── images/
│   └── split_zhou_Food101.json
├── oxford_flowers/
│   ├── jpg/
│   ├── imagelabels.mat
│   ├── cat_to_name.json
│   └── split_zhou_OxfordFlowers.json
├── oxford_pets/
│   ├── images/
│   ├── annotations/
│   └── split_zhou_OxfordPets.json
├── stanford_cars/
│   ├── cars_test/
│   └── split_zhou_StanfordCars.json
├── sun397/
│   ├── SUN397/
│   └── split_zhou_SUN397.json
└── ucf101/
    ├── UCF-101-midframes/
    └── split_zhou_UCF101.json
```

The CLI fails early if `--data-root` does not exist. Dataset-specific missing
files are reported by the DOTA data loaders.

# DEL-Analysis
This repository is built and designed to automate some of the tasks associated with the DEL Machine Learning (DEL ML) Pipeline. This is designed to work specifically with the Baylor DEL data that we will be receiving but could serve as some of the backbone for subsequent tools developed by our lab.

## Functionalities
Currently, this tool has the following scripts:
- active_overview.py -> Generates a general overview of actives for selection data files by providing the number of actives and the number of libraries containing these actives
- active_overlap.py -> Generates a Venn diagram of actives to generate a comprehensive view of the promiscuous compounds within these selections
- active_threshold.py -> Generates scatterplots for activity thresholds, these being based on z score and poly o enrichment metrics
- plot_tmap.py -> Generates a [tmap](https://github.com/reymond-group/tmap) representation of your selection data used for downstream ML purposes
- modelGen.py -> Generates simple Random Forest classification models that are optimized for [PPV](https://github.com/molecularmodelinglab/plate-ppv) metrics based on 5-fold cross validation

## Running the Scripts
These scripts will be designed to run through make that will help to streamline the process. To run these scripts, create conda environments from the envs/ directory. The DEL-Analysis environment will work run all pipeline processes, except for plot_tmap.py. This can be done through:

```bash
make
```

 For this, activate the tmap environment and run:
```bash
make Figures/tmap
``` 
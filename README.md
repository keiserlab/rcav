# Robust Semantic Interpretability: Revisiting Concept Activation Vectors

This repository contains the official pytorch implementation of [RCAV](https://linkhere.com) and the accompanying TFMNIST and Biased-CAMELYON16 datasets.

Robust Concept Activation Vectors (RCAV) quantifies the effects of semantic concepts on individual model predictions and on model behavior as a whole. By generalizing previous work on concept activation vectors to account for model non-linearity, and by introducing stricter hypothesis testing, we show that RCAV yields interpretations which are both more accurate at the image level and robust at the dataset level. RCAV, like saliency methods, supports the fine-grained interpretation of individual predictions.

The TFMNIST and B-CAMELYON16 datasets may be used as benchmarks for semantic interpretability methods.

### Run main.py to reproduce the results shown in Figure 2 of the paper. Note that main.py also accepts command line arguments, e.g. if you wish to retrain the model instead of loading the trained weights.

# Usage

main.py will save your reproduced copy of figure 2 to the file RCAV_fig2.png

rcav.py and rcav_utils.py contains the code for running RCAV on any model. 

Note use of rcav.py on another model requires adding latent augmentation functionality as is done in lines 150, 156, etc. of inception_mixup.py.

# Requirements: 
Please FIRST download model weights from https://zenodo.org/record/3889104 and put the file in the same directory as main.py

Requirements for these scripts may be installed by pip or conda using the requirements.txt or rcav_env.yml files.

# Datasets:
TFMNIST.py contains the code for creating the TFMNIST dataset note that the split is not the same as used for model training.
The B-CAMELYON16 dataset described in the paper will be made available shortly.
The unnaugmented data for CAMELYON16 is available at http://gigadb.org/dataset/view/id/100439/

# TODO:
Upload B-CAMELYON16

Upload split used for TFMNIST

Link RCAV to arxiv posting

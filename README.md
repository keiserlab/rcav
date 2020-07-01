Run main.py to reproduce the results shown in Figure 2 of the paper. Note that main.py also accepts command line arguments, e.g. if you wish to retrain the model instead of loading the trained weights.
main.py will save your reproduced copy of figure 2 to the file RCAV_fig2.png

Requirement: please FIRST download model weights from https://zenodo.org/record/3889104 and put the file in the same directory as main.py
Requirements for these scripts may be installed by pip or conda using the requirements.txt or rcav_env.yml files.

TFMNIST.py contains the code for creating the TFMNIST dataset.
rcav.py and rcav_utils.py contains the code for running RCAV on any model. 
Note use of rcav.py on another model requires adding latent augmentation functionality as is done in lines 150, 156, etc. of inception_mixup.py.

We plan to release code for the biased CAMELYON16 dataset publicly following review. For now, the unaugmented data for CAMELYON16 is available at http://gigadb.org/dataset/view/id/100439/
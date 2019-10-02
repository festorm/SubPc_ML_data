# SubPc database
The DFT database includes name, HOMO energies (HOMO), first TD-DFT excitation energies (E_opt), SMILES strings and one of the three utilized feature vectors (Coulomb, E-state and OneHot).

Data for structures with no TTF ligands are found in dssc_coulomb_nottf.pkl (Coulomb), estate_no_ttf.pkl (E-state index) and one_hot_no_ttf.pkl (OneHot). 

# Neural Network
Trained neural network algorithms are found in NN_XXX_HOMO_wd0001_hl3_hn10_lr001_3000e.pkl for HOMO energies and NN_XXX_LUMO_wd0001_hl3_hn10_lr001_3000e.pkl for LUMO energies with XXX = {coulomb, estate or onehot}.

Scripts for optimizing the hyperparameters with training/validition is given in XXXX. 

Script for training and testing the NN is given in NN_SubPc_HOMO_estate.py with the E-state feature vectors used as example. The E-state dataset has been split beforehand with seed 42. The corresponding data is given in: 

# Linear Regression
The script for linear regression is LinearReg.ipynb.

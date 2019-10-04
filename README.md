# SubPc database
The DFT database includes name, HOMO energies (HOMO), first TD-DFT excitation energies (E_opt), SMILES strings and one of the three utilized feature vectors (Coulomb, E-state and OneHot).

Data for structures with no TTF ligands are found in dssc_coulomb_nottf.pkl (Coulomb), estate_no_ttf.pkl (E-state index) and one_hot_no_ttf.pkl (OneHot). 

# Neural Network
Trained neural network (NN) algorithms are found in NN_XXX_HOMO_wd0001_hl3_hn10_lr001_3000e.pkl for HOMO energies and NN_XXX_LUMO_wd0001_hl3_hn10_lr001_3000e.pkl for LUMO energies with XXX = {coulomb, estate or onehot}.

Script for optimizing the hyperparameters with training/validition is given in NN_SubPC_HOMO_estate_opthyppar_crossval.py. The for loop (line 47) should be over the hyperparameter to be optimized, e.g., learning rate, epochs, weight decay parameter for the AdamW optimizer function, or number of hidden nodes (referred to as neurons in the script). Optimization of the number of hidden layers were done manually by changing the number of layers in the class definition of the NN.

Script for final training and testing the NN is given in NN_SubPc_HOMO_estate.py with the E-state feature vectors used as example. The E-state dataset has been split beforehand with seed 42. The corresponding data is given in: df_train_seed42.pkl, y_train_seed42.pkl, df_test_seed42.pkl and y_test_seed42.pkl.

# Linear Regression
The script for linear regression (LR) is LinearReg.ipynb. The final LR models are found in LRmodel_HOMO_XXX.sav and LRmodel_LUMO_XXX.sav for HOMO and LUMO energies, respectively, and with XXX = {coulomb, estate or onehot}.

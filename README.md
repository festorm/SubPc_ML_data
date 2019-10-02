# SubPc database
The DFT database includes name, HOMO energies (HOMO), first TD-DFT excitation energies (E_opt), SMILES strings and one of the three utilized feature vectors (Coulomb, E-state and OneHot).

Data for structures with no TTF ligands are found in
    - Coulomb: dssc_coulomb_nottf.pkl 
    - E-state index: estate_no_ttf.pkl
    - OneHot: one_hot_no_ttf.pkl 

# Neural Network
Trained neural network algorithms are found in NN_XXX_HOMO_wd0001_hl3_hn10_lr001_3000e.pkl for HOMO energies and NN_XXX_LUMO_wd0001_hl3_hn10_lr001_3000e.pkl for LUMO energies with XXX = {coulomb, estate or onehot}.

# Linear Regression
The script for linear regression is LinearReg.ipynb.

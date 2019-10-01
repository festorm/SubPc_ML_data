#!/home/koerstz/anaconda3/envs/rdkit-env/bin/python

"""
script for substituting the SubPc core in order to investigate the Donor/Acceptor character - E_gap

Author:
Mads Koerstz FEB/2018
Freja Storm JUNE/2018
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import itertools
import pandas as pd
import pickle

from multiprocessing import Pool

def perform_subtitution(packed):
    """
    Add ligand to the molecular core at atoms defined by their Idx.

    
    input
    ------
    packed : tuple((substituents, positions), parent mol)
        
        
    output
    -------
    smiles : tuple (number of ligands, smiles, ligands, position) smiles string of substituted core.        

    """
    global substituent_pattern # use dict to create filenames  

    sub_pat, mol = packed

    sub_mol = mol
    sub, pos = sub_pat
    #sub, pair = sub_pat
    #the list of all the substituents to investigate, and the possible positions where the substituents can be placed
    #pos = [x for x,y in pair]
    #h = [y for x,y in pair]

    num = mol.GetNumAtoms()

    num_sub = len(list(sub))
    #number of substituents to place on the SubPc molecule
    
    
    for j in range(num_sub):
        
        mod = Chem.MolFromSmiles(sub[j])
        
        mod_atoms = mod.GetNumAtoms()
        #modification
        
        combined = Chem.CombineMols(sub_mol, mod)
        # combine substituent and mol into one mol object
        
        # make a bond between core and ligand.
        ed_combined = Chem.EditableMol(combined)
        ed_combined.AddBond(pos[j],num,order=Chem.rdchem.BondType.SINGLE)
        #num is the first atom after the core atoms (index start at 0)


        sub_mol = ed_combined.GetMol()
        num = sub_mol.GetNumAtoms()
            
        #Chem.MolToMolFile(sub_mol,'mol.mol')

        #update the number of atoms if more substituents are needed
        
        #Chem.SanitizeMol(sub_mol)
        #m = Chem.MolFromSmiles(Chem.MolToSmiles(sub_mol))
        #m = Chem.AddHs(m)
        #AllChem.EmbedMolecule(m)
        #AllChem.UFFOptimizeMolecule(m)
        
        #print(Chem.MolToMolBlock(m))

    collect_sub = []
    
    for i in sub:
        collect_sub.append(substituent_pattern[i])

    #print ':'.join(map(str,collect_sub))
    #' '.join(map(str,pos))  
    name = str(num_sub) + '-' + ':'.join(map(str,collect_sub)) + '-' + ':'.join(map(str,pos))
    return (num_sub, name,Chem.MolToSmiles(sub_mol), ' '.join(map(str,sub)), ' '.join(map(str,pos)))


# Define the parent structure and the atoms to perform the substitutions.  
#mol = Chem.MolFromMolFile("subpc.mol",removeHs=False)
#mol = Chem.MolFromMolFile("subpc.mol")
mol=Chem.MolFromSmiles("[n+]12c3nc4n5c(nc6n(c(nc1c1c3cccc1)c1ccccc61)[B-]25Oc1ccc(cc1)C(C)(C)C)c1c4cccc1")

atoms = mol.GetNumAtoms()
for idx in range(atoms):
        mol.GetAtomWithIdx( idx ).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx() ))

        Draw.MolToFile(mol, "test.svg")


patt= Chem.MolFromSmarts('[cX3;H1]')  # smarts of atoms for substitution. All carbons bound to H

carbon_Idx = list() # atom idx list.
carbon2_Idx = list() # atom idx list.
#hydrogen_Idx = list()
for Idx in mol.GetSubstructMatches(patt):
    carbon_Idx.append(Idx[0])
    
    
    #if Idx[0] < 40:
    #    carbon_Idx.append(Idx[0])
        #hydrogen_Idx.append(Idx[1])
for i in [30,31,28,27]:
    carbon_Idx.remove(i)

print(carbon_Idx)

# list of ligands to substitute.
ligands = ['C#CC(F)(F)(F)',
           'C#CC(=O)[H]',
           'C#CC(=O)C',
           'C#CC(=O)N',
           'C#CS(=O)(=O)(C)',
           'C#CN',
           'C#CN(C)(C)',
           'C#CN(C(=O)(C))',
           'C#CC1=CC=C(F)C=C1',
           'C#CC1=CC=C(Cl)C=C1',
           'C#CC1=CC=C(Br)C=C1',
           'C#CC1=CC=C(C(F)(F)(F))C=C1',
           'C#CC1=CC=C(C#N)C=C1',
           'C#CC1=CC=C([N+](=O)([O-]))C=C1',
           'C#CC1=CC=C(C(=O)[H])C=C1',
           'C#CC1=CC=C(C(=O)O)C=C1',
           'C#CC1=CC=C(C(=O)C)C=C1',
           'C#CC1=CC=C(C(=O)N)C=C1',
           'C#CC1=CC=C(C#C)C=C1',
           'C#CC1=CC=C(S(=O)(=O)(C))C=C1',
           ]


#list of donor ligands
donors = ['C#CC1=CC=CC=C1',
          'C#CC1=CC=C(N)C=C1',
          r'C#CC1=C(C)S/C(S1)=C2SC(C)=C(C)S\2',
          'C#CN(C1=CC=CC=C1)C2=CC=CC=C2',
          'C#CO',
          'C#CNC(C)=O'
          ]


#list of acceptor ligands 
acceptors = ['C#CC1=CC=C(C(O)=O)C=C1',
             'C#CC1=CC=C([N+]([O-])=O)C=C1',
             'C#CC1=CC2=NSN=C2C=C1',
             'C#CC(O)=O',
             'C#CC#N',
             'C#CF'
             ]



#print(len(ligands))
#print(len(donors+acceptors))

# create dict
substituent_pattern = dict()
for i, sub in enumerate(ligands+donors+acceptors):
    substituent_pattern[sub] = i
#for i, sub in enumerate(zip(donors,acceptors)):
#    substituent_pattern[sub] = i

#for i, sub in enumerate(ligands):
#    substituent_pattern[sub] = i

pickle.dump(substituent_pattern, open('substituent_pattern.p', 'wb'))


substitutions = 2 
# how many substituents to maximally attach. #TODO create and acceptor and donor list for the doubly substiuted case

df = pd.DataFrame( ) # data frame to store results.

for i in range(1, substitutions+1):
    """
    Create an iterator that has all possible substitution patterns.
    """
 #   if i ==1:
    substituent_combinations = itertools.product(ligands+donors+acceptors, repeat=i) # All combinations of ligands (AA, AB, ...)
    pickle.dump(substituent_combinations, open('substituent_combinations.p','wb'))
 #   if i ==2:
 #       substituent_combinations = itertools.product(donors,acceptors, repeat=1) # All combinations of donor/acceptor ligands (AA, AB, ...)
     
    position_perturbations = itertools.permutations(carbon_Idx,i) # All the permutations of atom positions where the substitution can take place. For str of length i
    # the product of position permutations and ligand combination ((A,A), (1,1), ..)..
    substitution_pattern = itertools.product(substituent_combinations, position_perturbations)

    """
   # Run "substitution_pattern" in parallel. Put all results into a dataframe delete smiles duplicates.
    """
    
    packed_format = itertools.zip_longest(substitution_pattern, [], fillvalue=mol) # get the correct iterator for function.

    #perform_subtitution(list(packed_format)[0])
    pool = Pool(processes=5) # number of CPUs for parallel computing.
    results = pool.map(perform_subtitution, packed_format)

    df2 = pd.DataFrame(results, columns=['sub', 'name', 'smiles', 'ligand', 'position'])
    #df2.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
    
    print(df2.shape[0])

    df = df.append(df2) #create a collected dataframe with all the substituon patterns

# write data to .csv file.
df.to_csv('test_DA_sub_smiles.csv',index=False, header=False, sep=',', mode='w')

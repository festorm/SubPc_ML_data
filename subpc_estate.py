#!/groups/kemi/freja/anaconda3/envs/my-rdkit-env/bin/python

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem.EState import EStateIndices 
from rdkit.Chem.EState import AtomTypes 

from subprocess import Popen, PIPE
import os

import pickle

def finger_print(mol,name,e_opt): 
    """ 
    Create a dictionary with the e-state fingerprint for the molecule in mol (rdkit mol)
    
    Input:
    mol; rdkit mol object
    name; structure name
    e_opt; energy gap (target)
    """ 
    
    types = AtomTypes.TypeAtoms(mol) 
    es = EStateIndices(mol) 
    counts, sums = Fingerprinter.FingerprintMol(mol)
    
    if AtomTypes.esPatterns is None: 
        AtomTypes.BuildPatts() 
    
    name_list = [name for name,_ in AtomTypes.esPatterns]
    
    data={'name':name,'E_opt':e_opt}
    data2 = {k: v for k,v in zip(name_list,sums)}
    
    data.update(data2)
    return data

def write_xyz_file(Atoms, filename):
    num_atoms = len(Atoms)
    nitrogen_atoms = []
    with open(filename,"w") as f:
        f.write("{}\n \n".format(num_atoms))
        for i,coords in enumerate(Atoms):
            atom_type, x,y, z = coords
            x,y,z = float(x),float(y),float(z)
            if atom_type == "1":
                atom_type="H"
            elif atom_type == "5":
                b_num = i
                b_coords = [x,y,z]
                atom_type="B"
            elif atom_type == "6":
                atom_type="C"
            elif atom_type == "7":
                nitrogen_atoms.append([i,x,y,z])
                atom_type="N"
            elif atom_type == "8":
                atom_type="O"
            elif atom_type == "9": 
                atom_type="F"
            elif atom_type == "16": 
                atom_type="S"
            elif atom_type == "17": 
                atom_type="Cl"
            elif atom_type == "26": 
                atom_type="Fe"
            elif atom_type == "35": 
                atom_type="Br"
            f.write("{} \t {} \t {} \t {} \n".format(atom_type,x,y,z))
    for i in range(len(nitrogen_atoms)):
        distance = np.linalg.norm(np.array(b_coords) - np.array(nitrogen_atoms[i][1:]))
        if distance < 1.6:
            n_num = nitrogen_atoms[i][0]
            break
    return b_num, n_num

def create_sdf(filename,b_num,n_num):
    sdf = shell('obabel -ixyz '+filename+' -osdf',shell=True).split('\n')
    with open("mol.sdf","w") as f:
        for string in sdf:
            if "M  RAD" in string:
                string = "M  CHG  2   "+str(n_num)+"   1  "+str(b_num)+"  -1"
            f.write(string + '\n')

def create_smile(filename):
    """
    Create at .pdb file from xyz coordinates 
    TODO: do this without the shell 
    """
    smile = shell('obabel -ixyz '+filename+ ' -osmi ',shell=True).split('\n')
     
    smile = smile[0].split('\t')[0]
    return smile

def shell(cmd, shell=False):
    """ 
    runs the shell command cmd
    """
    if shell:
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    
    else:
        cmd = cmd.split()
        p = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    output, err = p.communicate()
    
    return output.decode('utf-8')


if AtomTypes.esPatterns is None: 
    AtomTypes.BuildPatts()

name_list = [name for name,_ in AtomTypes.esPatterns]

df2 = pd.DataFrame(columns=['name','E_opt']+name_list) #create the name, target, features 

df = pd.read_pickle("./egap_subpc.pkl")

for row_index,row in df.iterrows():
    #if row_index < 10:
    #if row_index < 1000:
    Atom = (row["Atom"])
    name = (row["name"])
    E_opt = (row["E_opt"])

    b_num,n_num = write_xyz_file(Atom, str(row_index) +'.xyz')
    #print(b_num,n_num)
    #create_sdf(str(row_index) +'.xyz',b_num,n_num)

    #m = Chem.SDMolSupplier("mol.sdf",removeHs=False, sanitize=False)
    #m = m[0]

    smile = create_smile(str(row_index) +'.xyz')
    #suggestion from https://sourceforge.net/p/rdkit/mailman/message/35835072/

    m = Chem.MolFromSmiles(smile,sanitize=False)
    m.UpdatePropertyCache(strict=False)

    Chem.SanitizeMol(m,Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_SETCONJUGATION|Chem.SANITIZE_SETHYBRIDIZATION)

    os.system('rm ' + str(row_index) +'.xyz')

#     with open("smile_strings.txt","a") as f:
#         f.write(smile + "\n")

    try:
        data= finger_print(m,name,E_opt)
        df2 = df2.append(data,ignore_index=True)

    except AttributeError:
        print(i,formula)
    continue

df2.to_pickle("./estate_subpc.pkl")


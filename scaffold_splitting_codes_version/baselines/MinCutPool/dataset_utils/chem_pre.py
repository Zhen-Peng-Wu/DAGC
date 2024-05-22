import torch
import rdkit.Chem as Chem
import numpy as np
import rdkit


def pre_filter(smile):
    """
    :param smile: molecular smile
    :return: flag to be kept or not (kept: True, not: False)
    """
    mol = Chem.MolFromSmiles(smile)

    mol_error_flag1 = check_degree(mol)

    mol_error_flag2 = check_max_atom_num(mol)

    if mol_error_flag1 == 0 & mol_error_flag2 == 0:
        return True
    else:
        return False


def check_degree(mol, max_degree=5):
    mol_error_flag = 0
    for atom in mol.GetAtoms():
        if atom.GetDegree() > max_degree:
            mol_error_flag = 1
            break
    return mol_error_flag


def check_max_atom_num(mol, max_atom_num=102):
    mol_error_flag = 0
    if len(mol.GetAtoms()) > max_atom_num:
        mol_error_flag = 1
    return mol_error_flag



def pre_transform(data, smile):
    """
    :param data: torch-geometric data
    :param smile: molecular smile
    :return: the featurized torch-geometric data
    """
    mol = Chem.MolFromSmiles(smile)

    AtomFeatureMat = torch.Tensor(GetMolFeatureMat(mol))
    data.x = AtomFeatureMat
    BondFeatureMat = torch.Tensor(GetBondFeatureMat(mol))
    data.edge_attr = BondFeatureMat
    data.num_tasks = len(data.y)
    return data

def GetAtomFeatures(atom):
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    SymbolList = ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At']
    if symbol in SymbolList:
        loc = SymbolList.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    HybridizationList = [rdkit.Chem.rdchem.HybridizationType.SP,
                         rdkit.Chem.rdchem.HybridizationType.SP2,
                         rdkit.Chem.rdchem.HybridizationType.SP3,
                         rdkit.Chem.rdchem.HybridizationType.SP3D,
                         rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in HybridizationList:
        loc = HybridizationList.index(hyb)
        feature[loc+24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31+hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        feature[36] = 1
        try:
            chi = atom.GetProp('_CIPCode')
            ChiList = ['R','S']
            loc = ChiList.index(chi)
            feature[37+loc] = 1
            #print("Chirality resolving finished.")
        except:
            feature[37] = 0
            feature[38] = 0
    return feature

def GetMolFeatureMat(mol):
    FeatureMat = []
    for atom in mol.GetAtoms():
        feature = GetAtomFeatures(atom)
        FeatureMat.append(feature.tolist())
    return FeatureMat

def GetBondFeatures(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    BondTypeList = [rdkit.Chem.rdchem.BondType.SINGLE,
                    rdkit.Chem.rdchem.BondType.DOUBLE,
                    rdkit.Chem.rdchem.BondType.TRIPLE,
                    rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in BondTypeList:
        loc = BondTypeList.index(type)
        feature[0+loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = conj

    # stereo
    stereo = bond.GetStereo()
    StereoList = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                  rdkit.Chem.rdchem.BondStereo.STEREOANY,
                  rdkit.Chem.rdchem.BondStereo.STEREOZ,
                  rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in StereoList:
        loc = StereoList.index(stereo)
        feature[6+loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature

def GetBondFeatureMat(mol, bidirection=False):
    FeatureMat = []
    for bond in mol.GetBonds():
        feature = GetBondFeatures(bond)
        FeatureMat.append(feature.tolist())
        if bidirection:
            FeatureMat.append(feature.tolist())

    # mol has no bonds
    if len(FeatureMat) == 0:
        FeatureMat = np.empty((0, 10), dtype = np.int64)
    else:
        FeatureMat = np.array(FeatureMat, dtype = np.int64)
    return FeatureMat


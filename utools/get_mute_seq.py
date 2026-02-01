import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Polypeptide
from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import write
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import re
import json
import sys
from torchmetrics.classification import BinaryAccuracy,BinaryAUROC
import torch
from math import fabs
from numpy import interp
from sklearn.metrics import roc_curve,average_precision_score,precision_recall_curve,auc
from munch import Munch
import math

sys.path.append('/home/u/data/xyh/project/interface_aware/')
from utils.cdr_extract import extract_CDR

def mutate_residue(structure, chain_id, res_id, new_res):
    model = structure[0]
    chain = model[chain_id]
    residue = chain[res_id]
    residue.resname = Polypeptide.one_to_three(new_res)
    return structure

# def get_chain_sequence(chain):
#     ppb = Polypeptide.PPBuilder()
#     for pp in ppb.build_peptides(chain):
#         return pp.get_sequence()

standard_amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
amino_acid_codes = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                    "TYR": "Y", "VAL": "V"}

def get_chain_sequence(structure,chain_id):
    model = structure[0]
    chain = model[chain_id]
    sequence = ""
    for residue in chain:
        if residue.get_resname() in standard_amino_acids:
            sequence += amino_acid_codes[residue.get_resname()]
    return sequence

def get_mute_seq(dataset):
    # ab_bind = pd.read_csv(f'/home/u/data/xyh/project/interface_aware/data/AB-Bind/process_data.csv',encoding='ISO-8859-1')
    # ab_bind = ab_bind[['#PDB','antigen','heavy','light','mutation','ddG']]
    # skmepi = pd.read_csv(f'/home/u/data/xyh/project/interface_aware/data/SKEMPI/process_data.csv',encoding='ISO-8859-1')
    # skmepi = skmepi[['#PDB', 'antigen', 'heavy', 'light', 'mutation', 'ddG']]
    # pd.concat([ab_bind])

    data = pd.read_csv(f'/home/u/data/xyh/project/interface_aware/data/{dataset}/process_data.csv',encoding='ISO-8859-1')
    error_index = set()
    error_list = []
    all_data = []
    for index,values in data.iterrows():
        row = []
        pdb_id,mutation,heavy,light,antigen,ddG= values['#PDB'],values['mutation'],values['heavy'],values['light'],values['antigen'],values['ddG']
        row+=[pdb_id,heavy,light,antigen,mutation]
        mutation = mutation.split(',')
        pdb_file = f'/home/u/data/xyh/project/interface_aware/data/{dataset}/{pdb_id}.pdb'
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        # 输入字符串
        # 使用正则表达式提取所有字母字符
        # if pdb_id not in all_enty:
        def structure_to_seq(structure):
            seq = ["","","",""]
            # heavy_chain = structure[0][heavy]
            h = get_chain_sequence(structure,heavy)
            seq[1] = h
            if pd.notna(light):
                # light_chain = structure[0][light]
                l= get_chain_sequence(structure,light)
                seq[0] = l
            for i,chain_id in enumerate(antigen.split(',')):
                # chain = structure[0][chain_id]
                # Get the original sequence
                sequence = get_chain_sequence(structure,chain_id)
                seq[i+2]= sequence
            return seq
        row += structure_to_seq(structure)
        #对结构进行突变
        for muta in mutation:
            #original_residue,chain_id,position_number,position_letter,new_residue
            match = re.match(r'([A-Z])([A-Z])(\d+)([a-z]?)([A-Z])', muta)
            chain_id = match.group(2)
            # original_residue = match.group(2)
            position_number = match.group(3)
            position_letter = match.group(4)
            new_residue = match.group(5)
            if position_letter != '':
                res_id = (' ', int(position_number), position_letter.upper())
            else:
                res_id = (' ', int(position_number), ' ')
            new_res = new_residue
            # print(pdb_id,mutation,chain_id,position_number,new_residue)
            structure = mutate_residue(structure, chain_id, res_id, new_res)
        row += structure_to_seq(structure)
        row.append(ddG)
        all_data.append(row)
    all_data = pd.DataFrame(all_data,columns=['PDB','heavy','light','antigen','Mutation', 'antibody_light_seq', 'antibody_heavy_seq', 'antigen_a_seq',
                               'antigen_b_seq', 'antibody_light_seq_mut', 'antibody_heavy_seq_mut', 'antigen_a_seq_mut',
                               'antigen_b_seq_mut', 'ddG'])
    all_data.to_csv(f'/home/u/data/xyh/project/interface_aware/data/{dataset}/mute_data.csv',index=False)
    print(f"Over {dataset}")


# get_mute_seq('AB-Bind')
# get_mute_seq('SKEMPI')

def merge_data():
    ab_bind = pd.read_csv(f'/home/u/data/xyh/project/interface_aware/data/AB-Bind/mute_data.csv',encoding='ISO-8859-1')
    skmepi = pd.read_csv(f'/home/u/data/xyh/project/interface_aware/data/SKEMPI/mute_data.csv',encoding='ISO-8859-1')
    data = pd.concat([ab_bind,skmepi])
    antibody_light_cdr = []
    antibody_height_cdr = []
    antibody_light_cdr_mut = []
    antibody_height_cdr_mut = []
    for index,values in data.iterrows():
        pdb = values['PDB']
        abls = values['antibody_light_seq']
        abhs = values['antibody_heavy_seq']

        abls_m = values['antibody_light_seq_mut']
        abhs_m = values['antibody_heavy_seq_mut']
        ab_info, ab_cdr = extract_CDR(abhs, abls)
        height_cdr = ab_info['H_cdr']
        if 'L_cdr' in ab_info:
            light_cdr = ab_info['L_cdr']
        else:
            light_cdr = ""
        ab_info_m, ab_cdr_m = extract_CDR(abhs_m, abls_m)
        height_cdr_m = ab_info_m['H_cdr']
        if 'L_cdr' in ab_info_m:
            light_cdr_m = ab_info_m['L_cdr']
        else:
            light_cdr_m = ""

        antibody_height_cdr.append(height_cdr)
        antibody_light_cdr.append(light_cdr)

        antibody_height_cdr_mut.append(height_cdr_m)
        antibody_light_cdr_mut.append(light_cdr_m)

    data.insert(loc=14,column='antibody_height_cdr',value=antibody_height_cdr)
    data.insert(loc=15,column='antibody_light_cdr',value=antibody_light_cdr)
    data.insert(loc=16,column='antibody_height_cdr_mut',value=antibody_height_cdr_mut)
    data.insert(loc=17,column='antibody_light_cdr_mut',value=antibody_light_cdr_mut)
    data.to_csv('/home/u/data/xyh/project/interface_aware/data/Merge_mutant/mute_data.csv',index=False)
# merge_data()


if __name__ == '__main__':
    get_mute_seq('AB-Bind')
    # get_mute_seq('SKEMPI')
    # merge_data()

    # pdb_file = f'/home/u/data/xyh/project/interface_aware/data/AB-Bind/1DQJ.pdb'
    # parser = PDBParser(PERMISSIVE=1)
    # structure = parser.get_structure('1DQJ', pdb_file)
    # # 输入字符串
    # # 使用正则表达式提取所有字母字符
    # # if pdb_id not in all_enty:
    # def structure_to_seq(structure):
    #     seq = ["","","",""]
    #     heavy,light,antigen = 'H','L','C'
    #     h = get_chain_sequence(structure,heavy)
    #     seq[1] = h
    #     if pd.notna(light):
    #         l= get_chain_sequence(structure,light)
    #         seq[0] = l
    #     for i,chain_id in enumerate(antigen.split(',')):
    #         # chain = structure[0][chain_id]
    #         # Get the original sequence
    #         sequence = get_chain_sequence(structure,chain_id)
    #         seq[i+2]= sequence
    #     return seq
    # seq = structure_to_seq(structure)
    # print(seq)
    # for muta in ['DH32A','KC97A']:
    #     # original_residue,chain_id,position_number,position_letter,new_residue
    #     match = re.match(r'([A-Z])([A-Z])(\d+)([a-z]?)([A-Z])', muta)
    #     chain_id = match.group(2)
    #     # original_residue = match.group(2)
    #     position_number = match.group(3)
    #     position_letter = match.group(4)
    #     new_residue = match.group(5)
    #     if position_letter != '':
    #         res_id = (' ', int(position_number), position_letter.upper())
    #     else:
    #         res_id = (' ', int(position_number), ' ')
    #     new_res = new_residue
    #     # print(pdb_id,mutation,chain_id,position_number,new_residue)
    #     structure = mutate_residue(structure, chain_id, res_id, new_res)
    # mu_seq = structure_to_seq(structure)
    # print(mu_seq)
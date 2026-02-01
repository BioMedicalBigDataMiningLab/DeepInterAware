import pandas as pd
from Bio import PDB
import re
from get_mute_seq import standard_amino_acids,amino_acid_codes
def process(pdb_id):
    """Recoding amino acid positions"""
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb_id, f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}.pdb")
    chains = structure.get_chains()
    for chain in chains:
        residues = chain.get_residues()
        new_residue_number = 1001
        # print(chain)
        for residue in residues:
            # print(residue.id,new_residue_number)
            residue.id = (' ', new_residue_number, ' ')
            new_residue_number += 1

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}_r.pdb")

    structure = parser.get_structure(pdb_id, f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}_r.pdb")
    chains = structure.get_chains()
    for chain in chains:
        residues = chain.get_residues()
        for residue in residues:
            _,residue_id,_ = residue.id
            residue.id = (' ', residue_id-1000, ' ')
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}_r.pdb")

def clean_pdb():
    data = pd.read_csv(f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/process_data.csv")
    #original_residue,chain_id,position_number,position_letter,new_residue
    clean_pdb_id = set()
    for index,values in data.iterrows():
        pdb_id,mutation = values['#PDB'],values['mutation']
        mutation = mutation.split(',')
        for muta in mutation:
            match = re.match(r'([A-Z])([A-Z])(\d+)([a-z]?)([A-Z])', muta)
            chain_id = match.group(2)
            # original_residue = match.group(2)
            position_number = match.group(3)
            position_letter = match.group(4)
            new_residue = match.group(5)
            if position_letter != '':
                clean_pdb_id.add(pdb_id)
                res_id = (' ', int(position_number), position_letter.upper())
    print(clean_pdb_id)
    #{'1N8Z', '3BE1', 'HM_3BN9', '3BDY', '3BN9', '2NY7'}
    for pdb_id in clean_pdb_id:
        process(pdb_id)

def compare(pdb_id):
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb_id, f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}.pdb")
    chains = structure.get_chains()
    structure_r = parser.get_structure(pdb_id, f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}_r.pdb")
    chains_r = structure_r.get_chains()
    map_dict = {}
    #
    # clean_process_data = clean_process_data[clean_process_data['#PDB']==pdb_id]
    # with open(f"/home/u/data/xyh/project/interface_aware/data/AB-Bind/{pdb_id}_r.txt",'w') as f:
    for chain,chain_r in zip(chains,chains_r):
        residues = chain.get_residues()
        residues_r = chain_r.get_residues()
        c_id = chain.id
        for residue,residue_r in zip(residues,residues_r):
            # print(residue.id,new_residue_number)
            if residue.get_resname() in standard_amino_acids:
                res = amino_acid_codes[residue.get_resname()]
                _,res_number,res_letter = residue.id
                key = (c_id+str(res_number)+res_letter.lower()).strip()

                res_r = amino_acid_codes[residue_r.get_resname()]
                _, res_number_r, res_letter_r = residue_r.id
                value = (c_id + str(res_number_r) + res_letter_r.lower()).strip()
                map_dict[key] = value
        #             f.write(f'{chain.id} {res} {residue.id} {res_r} {residue_r.id}\n')
        # f.close()
    return map_dict
def clean_data():
    all_map = {}
    for pdb_id in ['1N8Z', '3BE1', 'HM_3BN9', '3BDY', '3BN9', '2NY7']:
        map_dict = compare(pdb_id)
        all_map[pdb_id] = map_dict
        # print(all_map)
    clean_process_data = pd.read_csv('/home/u/data/xyh/project/interface_aware/data/AB-Bind/clean_process_data.csv')
    for index,values in clean_process_data.iterrows():
        pdb_id,mutation = values['#PDB'],values['mutation']
        if pdb_id not in all_map:continue
        new_mutation = []
        for m in mutation.split(','):
            # print(m[1:-1])
            res = m[0]
            mu_res = m[-1]
            m = res+all_map[pdb_id][m[1:-1]]+mu_res
            new_mutation.append(m)
        new_mutation = ",".join(new_mutation)
        clean_process_data['mutation'].iloc[index] = new_mutation
    clean_process_data.to_csv('/home/u/data/xyh/project/interface_aware/data/AB-Bind/new_clean_process_data.csv',index=False)
clean_data()

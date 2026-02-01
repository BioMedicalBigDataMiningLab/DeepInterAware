import json

import pandas as pd
import subprocess
import math
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score
import scipy
import numpy as np
def skempi():
    data = pd.read_csv('/home/u/data/xyh/project/interface_aware/data/SKEMPI/process_data.csv')
    all_dict = {}
    all_id = []
    for index,values in data.iterrows():
        pdb_id,mutation = values['#Pdb'],values['Mutation(s)_cleaned']
        id = pdb_id.split('_')[0]
        # print(id, type(mutation))
        all_id.append(id)
        if id in all_dict:
            all_dict[id].append(mutation+';')
        else:
            all_dict[id] = [mutation + ';']
    data.insert(loc=0,value=all_id,column='pdb')
    data.to_csv('/home/u/data/xyh/project/interface_aware/data/SKEMPI/process_data.csv',index=False)
    for key,values in all_dict.items():
        with open(f'/home/u/data/xyh/project/interface_aware/data/SKEMPI/{key}.txt','w') as f:
            for i in values:
                f.write(i+'\n')
            f.close()

def ab_bind():
    data = pd.read_csv('/home/u/data/xyh/project/interface_aware/data/AB-Bind/process_data.csv')
    all_dict = {}
    mu = []
    for index, values in data.iterrows():
        id, mutation = values['#PDB'], values['Mutation']
        mutation = mutation.split(',')
        s = []
        for i in mutation:
            chain = i[0]
            AA = i[2]
            res = i[3:]
            s.append(AA+chain+res)
        # print(id, type(mutation))
        s = ",".join(s)
        mu.append(s)
        if id in all_dict:
            all_dict[id].append(s + ';')
        else:
            all_dict[id] = [s + ';']
    data.insert(loc=4,column='mutation',value=mu)
    data.to_csv('/home/u/data/xyh/project/interface_aware/data/AB-Bind/process_data.csv',index=False)
    for key, values in all_dict.items():
        with open(f'/home/u/data/xyh/project/interface_aware/data/AB-Bind/{key}.txt', 'w') as f:
            for i in values:
                f.write(i + '\n')
            f.close()




def deal(pdb_id,dataset):
    command = ['run_SSIPe.pl', '-pdb', f'{dataset}/{pdb_id}.pdb', '-mulist', f'{dataset}/{pdb_id}.txt', '-forcefield', 'EVOEF',
               '-isscore', 0.5, '-output', f'{dataset}/{pdb_id}_result.txt']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout
#res 会将deal返回的结果自动拼接
# def run_ab_bind():
#     data = pd.read_csv('/home/u/data/xyh/project/SSIPe/AB-Bind/process_data.csv')
#     pdb_list = set()
#     for index, values in data.iterrows():
#         id = values['#PDB']
#         pdb_list.add(id)
#     res=Parallel(n_jobs=8)(delayed(deal)(id) for id in pdb_list)

def run_ab_bind(dataset):
    data = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/{dataset}/process_data.csv')
    pdb_list = set()
    for index, values in data.iterrows():
        id = values['#PDB']
        pdb_list.add(id)
    for id in pdb_list:
        deal(id,dataset)


def process_ab_bind():
    data = pd.read_csv('/home/u/data/xyh/project/interface_aware/data/AB-Bind/process_data.csv')
    pdb_list = set()
    for index, values in data.iterrows():
        id = values['#PDB']
        pdb_list.add(id)
    result = []
    for id in pdb_list:
        temp = []
        with open(f'/home/u/data/xyh/project/SSIPe/AB-Bind/{id}_result.txt','r') as f:
            for lines in f.readlines()[1:]:
                if lines !=None:
                    _,EvoEF,mutations = lines.split()
                    temp.append([id,float(EvoEF),mutations.strip()[:-1]])
        # data = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/AB-Bind/{id}_result.txt',sep=' ')
        temp = pd.DataFrame(temp,columns=['pdb','EvoEF','Mutation'])
        result.append(temp)
    result = pd.concat(result)
    # print(result)
    ddG = []
    for index, values in result.iterrows():
        pdb,mutations = values['pdb'],values['Mutation']
        # print(pdb,mutations)
        filter = data[(data['#PDB']==pdb) & (data['mutation']==mutations)]

        ddG.append(filter['ddG(kcal/mol)'].values[0])
    result.insert(loc=1,column='ddG',value=ddG)
    result.to_csv(f'/home/u/data/xyh/project/SSIPe/AB-Bind/result.csv',index=False)

def process_skempi():
    data = pd.read_csv('/home/u/data/xyh/project/interface_aware/data/SKEMPI/process_data.csv')
    pdb_list = set()
    for index, values in data.iterrows():
        pdb_id, mutation = values['#Pdb'], values['Mutation(s)_cleaned']
        id = pdb_id.split('_')[0]
        pdb_list.add(id)
    result = []
    for id in pdb_list:
        temp = []
        with open(f'/home/u/data/xyh/project/SSIPe/SKEMPI/{id}_result.txt','r') as f:
            for lines in f.readlines()[1:]:
                if lines !=None:
                    _,EvoEF,mutations = lines.split()
                    temp.append([id,float(EvoEF),mutations.strip()[:-1]])
        # data = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/AB-Bind/{id}_result.txt',sep=' ')
        temp = pd.DataFrame(temp,columns=['pdb','EvoEF','Mutation'])
        result.append(temp)
    result = pd.concat(result)
    # print(result)
    ddG = []
    for index, values in result.iterrows():
        pdb,mutations = values['pdb'],values['Mutation']
        filter = data[(data['pdb']==pdb) & (data['Mutation(s)_cleaned']==mutations)]
        if filter.shape[0]>1:
            filter = filter[filter['ddG']!=0]
            if filter.shape[0]==0:
                ddg = 0
            else:

                ddg = filter['ddG'].values.mean()
        else:
            ddg = filter['ddG'].values[0]
        # print(filter)
        ddG.append(ddg)
    result.insert(loc=1,column='ddG',value=ddG)
    result.to_csv(f'/home/u/data/xyh/project/SSIPe/SKEMPI/result.csv',index=False)

# ab_bind()
# skempi()
# process_skempi()
def deal_foldx(pdb_id,dataset):
    #进入foldx文件夹
    command = f'./foldx_20241231 --command=BuildModel --pdb-dir=../{dataset} --pdb={pdb_id}.pdb --mutant-file=./{dataset}/individual_list_{pdb_id}.txt --output-dir=./AB-Bind_output'
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    return result.stdout
def run_foldx(dataset):
    data = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/{dataset}/process_data.csv')
    all_dict = {}
    all_id = []
    for index, values in data.iterrows():
        id, mutation = values['#PDB'], values['mutation']
        all_id.append(id)
        if id in all_dict:
            all_dict[id].append(mutation + ';')
        else:
            all_dict[id] = [mutation + ';']

    for key, values in all_dict.items():
        with open(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}/individual_list_{key}.txt', 'w') as f:
            for i in values:
                f.write(i + '\n')
            f.close()
            deal_foldx(key, dataset)

def process_foldx(dataset):
    data = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}/process_data.csv')
    pdb_list = data['#PDB'].drop_duplicates()
    all_result = {}
    for id in pdb_list:
        with open(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}_output/Dif_{id}.fxout','r') as f:
            difs = f.readlines()[9:]
        with open(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}/individual_list_{id}.txt','r') as f:
            mutations =  f.readlines()
        for mutation,dif in zip(mutations,difs):
            mutation = mutation[:-2] #去掉';'
            ddg = dif.split()[1] #ddg
            key = id+'_'+mutation
            all_result[key] = float(ddg)
    with open(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}/foldx.json','w') as f:
        json.dump(all_result,f,indent=4)
    result = []
    for index,values in data.iterrows():
        pdb_id,mutation,ddG = values['#PDB'],values['mutation'],values['ddG']
        key = pdb_id+'_'+mutation
        fold_ddG = all_result[key]
        result.append([pdb_id,mutation,ddG,fold_ddG])
    result = pd.DataFrame(result,columns=['PDB','Mutation','ddG','FoldX'])
    result.to_csv(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}/foldx_result.csv',index=False)



def clean_ab_bind():
    data = pd.read_csv('/home/u/data/xyh/project/interface_aware/data/AB-Bind/clean_process_data.csv')
    all_dict = {}
    for index, values in data.iterrows():
        id, mutation = values['#PDB'], values['mutation']
        if id in all_dict:
            all_dict[id].append(mutation + ';')
        else:
            all_dict[id] = [mutation + ';']
    for key, values in all_dict.items():
        with open(f'/home/u/data/xyh/project/SSIPe/Foldx/AB-Bind/individual_list_{key}.txt', 'w') as f:
            for i in values:
                f.write(i + '\n')
            f.close()

# process_foldx('SKEMPI')
def calculate_result(dataset):
    e_RMSE_whole,e_PCC_whole,e_MAE_whole,e_R2_whole = [],[],[],[]
    f_RMSE_whole,f_PCC_whole,f_MAE_whole,f_R2_whole = [],[],[],[]
    e_label,e_pred = [],[]
    f_label,f_pred = [],[]
    for fold in range(1,11):
        index = np.load(f'/home/u/data/xyh/project/interface_aware/data/{dataset}/val_index_fold_{fold}.npy')
        evoef = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/{dataset}/result.csv')
        foldx = pd.read_csv(f'/home/u/data/xyh/project/SSIPe/Foldx/{dataset}/foldx_result.csv')

        evoef = evoef.loc[index]
        print(evoef.shape)
        foldx = foldx.loc[index]
        print(foldx.shape)
        # ab_bind = pd.read_csv('/home/u/data/xyh/project/SSIPe/Foldx/AB-Bind/foldx_result.csv')
        # skempi = pd.read_csv('/home/u/data/xyh/project/SSIPe/Foldx/SKEMPI/foldx_result.csv')
        e_y_label = evoef['ddG'].values
        e_y_pred = evoef['EvoEF'].values
        e_label.append(e_y_label)
        e_pred.append(e_y_pred)
        # y_pred = data['FoldX'].values
        f_y_label = foldx['ddG'].values
        f_y_pred = foldx['FoldX'].values
        f_label.append(f_y_label)
        f_pred.append(f_y_pred)

    e_label, e_pred = np.concatenate(e_label), np.concatenate(e_pred)
    f_label, f_pred = np.concatenate(f_label), np.concatenate(f_pred)

    e_RMSE = np.sqrt(mean_squared_error(e_label, e_pred))
    e_PCC = scipy.stats.pearsonr(e_label, e_pred)[0]
    e_MAE = mean_absolute_error(e_label, e_pred)
    e_R2 = r2_score(e_label, e_pred)

    f_RMSE = np.sqrt(mean_squared_error(f_label, f_pred))
    f_PCC = scipy.stats.pearsonr(f_label, f_pred)[0]
    f_MAE = mean_absolute_error(f_label, f_pred)
    f_R2 = r2_score(f_label, f_pred)


    print("EvoEF")
    print(f"PCC {e_PCC} RMSE {e_RMSE} MAE {e_MAE} R2 {e_R2}")
    # y_pred = data['FoldX'].values
    print("FoldX")
    print(f"PCC {f_PCC} RMSE {f_RMSE} MAE {f_MAE} R2 {f_R2}")


def mute_result(model_name):
    data = pd.read_csv(f'/root/deepinteraware/result/mutant_result/{model_name}/all_fold_result.csv')
    print(data.mean())

# calculate_result('AB-Bind')
# calculate_result('SKEMPI')
# mute_result('AttABseq')
# calculate_result()
# data = pd.read_csv('/home/u/data/xyh/project/interface_aware/result/mutant_result/Rosetta/SKEMPI_result.csv')
# y_label = data['ddG'].values
# # y_pred = data['EvoEF'].values
# y_pred = data['Rosetta'].values
# RMSD = np.sqrt(mean_squared_error(y_label, y_pred))
# PCC = scipy.stats.pearsonr(y_label, y_pred)[0]
# MAE = mean_absolute_error(y_label, y_pred)
# R2 = r2_score(y_label, y_pred)
# print(f"PCC = {PCC} RMSE = {RMSD} MAE = {MAE} R2 = {R2}")
# clean_ab_bind()
# run_foldx('AB-Bind')
# run_foldx('SKEMPI')
# all_data = []
# for fold in range(1,11):
#     data = pd.read_csv(f'/root/deepinteraware/result/mutant_result/DeepInterAware/SKEMPI/seed_3_fold_{fold}_result.csv')
#     all_data.append(data)
# all_data = pd.concat(all_data)
# all_data  = all_data[['y_label','y_pred']]
# all_data.to_csv('/root/deepinteraware/result/mutant_result/DeepInterAware/SKEMPI/draw.csv',index=False)
# for seed in range(5):
#     data = pd.read_csv(f'/root/deepinteraware/result/mutant_result/DeepInterAware/AB-Bind/seed_{seed}_all_fold_metric_result_0.0005_32.csv')
#     print(data.mean())
data = pd.read_csv(f'/home/u/data/xyh/project/interface_aware/result/mutant_result/AttABseq/SKEMPI/all_result.csv')
print(data.mean())
# calculate_result('AB-Bind')
# calculate_result('SKEMPI')


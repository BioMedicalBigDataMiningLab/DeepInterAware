import pandas as pd
from biopandas.pdb import PandasPdb
import requests
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import os

from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryRecall
import os
import sys
import json
from munch import Munch

# from inference import data_norm_sigmoid

sys.path.append(os.getcwd())
from utools.cdr_extract import extract_CDR, get_CDR_simple
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy,BinaryAUROC,BinaryAveragePrecision,BinaryMatthewsCorrCoef,BinaryF1Score,BinaryPrecision,BinaryRecall
res_codes = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E',
    'PHE':'F','GLY':'G','HIS':'H','LYS':'K',
    'ILE':'I','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S',
    'THR':'T','VAL':'V','TYR':'Y','TRP':'W'}

def dis_pairs(coord_1,coord_2):
    coord_1_x = coord_1[-3]
    coord_1_y = coord_1[-2]
    coord_1_z = coord_1[-1]
    coord_2_x = coord_2[-3]
    coord_2_y = coord_2[-2]
    coord_2_z = coord_2[-1]
    distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + (float(coord_1_y) - float(coord_2_y)) ** 2 + (float(coord_1_z) - float(coord_2_z)) ** 2)
    return distance

#标记label
def get_labels(coord_AG,coord_H,coord_L,coord_H_res_id,coord_L_res_id,coord_AG_res_id):
    label_AG = [0 for i in range(len(coord_AG_res_id))]
    label_H = [0 for i in range(len(coord_H_res_id))]
    label_L = [0 for i in range(len(coord_L_res_id))]
    source_AGH = []
    target_AGH = []
    source_AGL = []
    target_AGL = []
    for i in range(len(coord_AG)):
        for j in range(len(coord_H)):
            # if(dis_pairs(coord_AG[i],coord_H[j]) <= 4.5):
            if(dis_pairs(coord_AG[i],coord_H[j]) <= 4.5):
                label_AG[coord_AG_res_id.index(coord_AG[i][0])] = 1
                label_H[coord_H_res_id.index(coord_H[j][0])] = 1
                if(coord_AG_res_id.index(coord_AG[i][0]) in source_AGH and coord_H_res_id.index(coord_H[j][0]) in target_AGH):
                    continue
                else:
                    source_AGH.append(coord_AG_res_id.index(coord_AG[i][0]))
                    target_AGH.append(coord_H_res_id.index(coord_H[j][0]))
        for k in range(len(coord_L)):
            # if(dis_pairs(coord_AG[i],coord_L[k]) <= 4.5):
            if(dis_pairs(coord_AG[i],coord_L[k]) <= 5.0):
                label_AG[coord_AG_res_id.index(coord_AG[i][0])] = 1
                label_L[coord_L_res_id.index(coord_L[k][0])] = 1
                if (coord_AG_res_id.index(coord_AG[i][0]) in source_AGL and coord_L_res_id.index(coord_L[k][0]) in target_AGL):
                    continue
                else:
                    source_AGL.append(coord_AG_res_id.index(coord_AG[i][0]))
                    target_AGL.append(coord_L_res_id.index(coord_L[k][0]))
    label_AGH = [source_AGH, target_AGH]
    label_AGL = [source_AGL, target_AGL]

    return label_AG,label_H,label_L,label_AGH,label_AGL

def coord(ATOM,Hchain,Lchain,AGchain):

    coord_H = []    # 存储原子坐标
    coord_L = []
    coord_AG = []
    coord_H_res_id=["FIRST"]
    coord_L_res_id=["FIRST"]
    coord_AG_res_id=["FIRST"]
    coord_H_res=[]
    coord_L_res =[]
    coord_AG_res =[]
    res_id_before="none"
    # 提前记录所有残基的去重id
    res_id_H=[]
    res_id_L=[]
    res_id_AG=[]
    for row in range(ATOM.shape[0]):
        """遍历所有原子标记原子氨基酸"""
        row_info = (np.array(ATOM.iloc[row, :]).tolist())
        res_id = row_info[7] + str(row_info[8]) + str(row_info[5])  # 该残基的独一无二标记
        # ['ATOM', 4148, '', 'OXT', '', 'VAL', '', 'H', 225, '', '', 87.753, 61.481, -11.977, 1.0, 87.78, '', '', 'O', nan, 4650]
        if (row_info[7] == Hchain):
            if (row_info[3] == "CA" or row_info[3] == "CB"):
                res_id_H.append(res_id)
        elif (row_info[7] == Lchain):
            if (row_info[3] == "CA" or row_info[3] == "CB"):
                res_id_L.append(res_id)
        elif (row_info[7] in AGchain):
            if (row_info[3] == "CA" or row_info[3] == "CB"):
                res_id_AG.append(res_id)
    res_id_H = list(set(res_id_H))
    res_id_L = list(set(res_id_L))
    res_id_AG = list(set(res_id_AG))
    # print("res_id_L:",res_id_L)

    for row in range(ATOM.shape[0]):
        row_info=(np.array(ATOM.iloc[row,:]).tolist())
        # ['ATOM', 4148, '', 'OXT', '', 'VAL', '', 'H', 225, '', '', 87.753, 61.481, -11.977, 1.0, 87.78, '', '', 'O', nan, 4650]
        res_id = row_info[7] + str(row_info[8]) + str(row_info[5])  # 该残基的独一无二标记
        res_x = row_info[11]
        res_y = row_info[12]
        res_z = row_info[13]
        tag_coord_H_res_i = 0  # 标注
        tag_coord_L_res_i = 0
        tag_coord_AG_res_i = 0
        try:
            if (row_info[7] == Hchain):
                # 记录基本信息
                res_n = res_codes[row_info[5]]  # 残基名称
                # 属于重链
                if(res_id in res_id_H):
                    coord_H.append([res_id,res_n, res_x, res_y, res_z]) # 原子坐标
                    # if(res_id not in coord_H_res_id and coord_H_res_id[-1] != res_id):
                    if (coord_H_res_id[-1] != res_id):
                        coord_H_res_id.append(res_id)
                    if (row_info[3] == "CA" or row_info[3] == "CB"):
                        coord_H_res_i=[res_x, res_y, res_z]
                        tag_coord_H_res_i = 1

            if (row_info[7] == Lchain):
                # 记录基本信息
                res_n = res_codes[row_info[5]]  # 残基名称
                # 属于轻链
                if (res_id in res_id_L):
                    coord_L.append([res_id,res_n, res_x, res_y, res_z])
                    # if (res_id not in coord_L_res_id):
                    if (coord_L_res_id[-1] != res_id):
                        coord_L_res_id.append(res_id)
                    if (row_info[3] == "CA" or row_info[3] == "CB"):
                        coord_L_res_i = [res_x, res_y, res_z]
                        tag_coord_L_res_i = 1

            if (row_info[7] in AGchain):
                # 记录基本信息
                res_n = res_codes[row_info[5]]  # 残基名称
                # 属于抗原
                if (res_id in res_id_AG):
                    coord_AG.append([res_id,res_n, res_x, res_y, res_z])
                    # if (res_id not in coord_AG_res_id and coord_AG_res_id[-1] != res_id):
                    if (coord_AG_res_id[-1] != res_id):
                        coord_AG_res_id.append(res_id)
                    if (row_info[3] == "CA" or row_info[3] == "CB"):
                        coord_AG_res_i = [res_x, res_y, res_z]
                        tag_coord_AG_res_i = 1
        except:
            continue
        #更新res坐标
        if(res_id_before != res_id):
            if (tag_coord_AG_res_i == 1):
                coord_AG_res.append(coord_AG_res_i)
                res_id_before = res_id
            elif (tag_coord_H_res_i == 1):
                # print(res_id)
                coord_H_res.append(coord_H_res_i)
                res_id_before = res_id
            elif (tag_coord_L_res_i == 1):
                coord_L_res.append(coord_L_res_i)
                res_id_before = res_id
                # print(res_id)

    coord_H_res_id=coord_H_res_id[1:]
    coord_L_res_id = coord_L_res_id[1:]
    coord_AG_res_id = coord_AG_res_id[1:]

    if(len(coord_H_res) != len(coord_H_res_id)):
        print("complex_dict['coord_H'] ERROR!")
    if (len(coord_L_res) != len(coord_L_res_id)):
        print("complex_dict['coord_L'] ERROR!")
    if (len(coord_AG_res) != len(coord_AG_res_id)):
        print("complex_dict['coord_AG'] ERROR!")
    # print("===============coord_L_res_id===============")
    # for i in range(len(coord_L_res_id)):
    #     print(coord_L_res_id[i])

    return coord_H,coord_L,coord_AG,coord_H_res_id,coord_L_res_id,coord_AG_res_id,coord_H_res,coord_L_res,coord_AG_res

def id2fasta2dict(pdb_id,Hchain,Lchain,AGchain,pdb_file):
    try:
        ATOM=PandasPdb().read_pdb(f'{pdb_file}{pdb_id}.pdb').df['ATOM']
    except:
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        # 发起HTTP请求并下载PDB文件
        response = requests.get(pdb_url)

        # 检查请求是否成功
        if response.status_code == 200:
            # 构建PDB文件的本地路径
            pdb_filename = f"{os.getcwd()}/data/SAbDab/pdb/{pdb_id}.pdb"

            # 写入PDB文件到本地
            with open(pdb_filename, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded {pdb_id}.pdb")
            ATOM = PandasPdb().read_pdb(f'/autodl-tmp/DeepInterAware-main/data/SabDab/pdb/{pdb_id}.pdb').df['ATOM']
        else:
            pdb_url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_id}/?raw=true"

            # 发起HTTP请求并下载PDB文件
            response = requests.get(pdb_url)
            if response.status_code == 200:
                pdb_filename = f"/autodl-tmp/DeepInterAware-main/data/SabDab/pdb/{pdb_id}.pdb"

                # 写入PDB文件到本地
                with open(pdb_filename, 'wb') as f:
                    f.write(response.content)

                print(f"Downloaded {pdb_id}.pdb")
                ATOM = PandasPdb().read_pdb(f'/autodl-tmp/DeepInterAware-main/data/SabDab/pdb/{pdb_id}.pdb').df['ATOM']
            else:
                print(f"Downloaded {pdb_id} error")
                ppdb = PandasPdb().fetch_pdb(pdb_id)
                ATOM=ppdb.df['ATOM']
    Hchain=Hchain
    Lchain=Lchain
    AGchain=AGchain
    coord_H, coord_L, coord_AG, coord_H_res_id, coord_L_res_id, coord_AG_res_id, coord_H_res, coord_L_res, coord_AG_res=coord(ATOM,Hchain,Lchain,AGchain)
    res_H=list(map(lambda x: res_codes[x[-3:]],coord_H_res_id))
    res_L=list(map(lambda x: res_codes[x[-3:]],coord_L_res_id))
    res_AG=list(map(lambda x: res_codes[x[-3:]],coord_AG_res_id))
    res_H=''.join(res_H)
    res_L=''.join(res_L)
    res_AG=''.join(res_AG)
    label_AG,label_H,label_L,label_AGH,label_AGL=get_labels(coord_AG,coord_H,coord_L,coord_H_res_id,coord_L_res_id,coord_AG_res_id)

    paratope = label_H+label_L
    #创建字典
    full_seq = {"res_H":res_H,"res_L":res_L,"res_AG":res_AG}
    full_label = {"label_H":label_H,"label_L":label_L,
                  "epitope":label_AG,"paratope":paratope,'label_AGH':label_AGH,'label_AGL':label_AGL}

    return full_seq,full_label

def site_process(res_H,label_H,res_L,label_L):
    sites = []
    ab_info,ab_cdr = extract_CDR(res_H,res_L)
    if res_H != None and res_H !="":
        h_site = []
        for i in range(1,4):
            cdr_index, cdr_range = ab_info[f'H_cdr{i}_range']
            h_site += label_H[cdr_index:cdr_range]
        sites += h_site

    if res_L != None and res_L != "":
        l_site = []
        for i in range(1,4):
            cdr_index, cdr_range = ab_info[f'L_cdr{i}_range']
            l_site += label_L[cdr_index:cdr_range]
        sites += l_site

    return sites,ab_info,ab_cdr

def process_interaction_pattern(ab_info,label_AGH,label_AGL):
    def process_chain(chain):
        if chain == 'H':
            pair_label = label_AGH
        else:
            pair_label = label_AGL
        new_pair_label = []
        cdr1_start, cdr1_end = ab_info[f'{chain}_cdr1_range']
        cdr2_start, cdr2_end = ab_info[f'{chain}_cdr2_range']
        cdr3_start, cdr3_end = ab_info[f'{chain}_cdr3_range']

        cdr1_len = cdr1_end - cdr1_start
        cdr2_len = cdr2_end - cdr2_start
        # cdr3_len = cdr3_end - cdr3_start

        for i, j in zip(pair_label[0], pair_label[1]):
            if j >= cdr1_start and j <= cdr1_end:
                j = j - cdr1_start
            elif j >= cdr2_start and j <= cdr2_end:
                j = j - cdr2_start+cdr1_len
            elif j >= cdr3_start and j <= cdr3_end:
                j = j - cdr3_start+cdr1_len+cdr2_len
            else:
                #不在CDR区
                continue
            if chain == 'H':
                new_pair_label.append((j, i))
            else:
                new_pair_label.append((len_h_cdr+j, i))
        return new_pair_label
    if 'H_cdr' in ab_info:
        len_h_cdr = len(ab_info['H_cdr'])
        new_label_AGH = process_chain('H')
    else:
        len_h_cdr = 0
        new_label_AGH = []

    if 'L_cdr' in ab_info:
        new_label_AGL = process_chain('L')
        pair_label = new_label_AGH + new_label_AGL
    else:
        pair_label = new_label_AGH

    pair_label.sort(key=lambda x:x[0])
    full_pair_label = []
    for i, j in zip(label_AGH[0], label_AGH[1]):
        full_pair_label.append((i,j))
    for i, j in zip(label_AGL[0], label_AGL[1]):
        full_pair_label.append((i,j))
    full_pair_label.sort(key=lambda x: x[0])

    return full_pair_label,pair_label

def get_binding_site(pdb_id,heavy_chain,light_chain,antigen_chain,pdb_file='/root/autodl-tmp/DeepInterAware-main/data/SAbDab/pdb/'):
    full_seq,full_label = id2fasta2dict(pdb_id,heavy_chain,light_chain,antigen_chain,pdb_file) #(Ag,H),(Ag,L),坐标从0开始
    # print(full_seq)
    # print(full_label)
    ab_sites,ab_info,ab_cdr = site_process(full_seq['res_H'],full_label['label_H'],full_seq['res_L'],full_label['label_L'])
    # print(ab_info)
    # print(sites)
    full_pair_label,pair_label = process_interaction_pattern(ab_info, full_label['label_AGH'], full_label['label_AGL'])
    label_dict = {
        'epitope':full_label['epitope'],
        'paratope':ab_sites,
        'paratope-epitope':pair_label
    }
    seq_dict = {
        # 'H_cdr':ab_info['H_cdr'],
        'ag_seq':full_seq['res_AG'],
        'ab_cdr':ab_cdr
    }
    if 'H_cdr' in ab_info:
        seq_dict['H_cdr']=ab_info['H_cdr']
    if 'L_cdr' in ab_info:
        seq_dict['L_cdr']=ab_info['L_cdr']

    full_label['paratope-epitope'] = full_pair_label
    # print(seq_dict)
    # print(label_dict)
    return full_seq,full_label,seq_dict,ab_info,label_dict



def norm(att):
    att_mean = torch.mean(att, dim=-1, keepdim=True)
    att_std  = torch.std(att, dim=-1, keepdim=True)
    att_norm = (att - att_mean) / att_std
    return att_norm

def calculate_binding_pair_recall(filter_att,pair_label,ag_len,ab_len,threshold):
    att_label = torch.zeros(ab_len,ag_len)
    for i, j in pair_label:
        att_label[i][j] = 1

    att = []
    for i in filter_att:
        att.append(norm(i))
    filter_att = torch.stack(att)
    filter_att = torch.sigmoid(filter_att).detach().to('cpu')
    filter_att = torch.ceil(filter_att - threshold).long()

    # 计算 True Positives (TP) 和 False Negatives (FN)
    TP = torch.sum((att_label == 1) & (filter_att == 1))
    FN = torch.sum((att_label == 1) & (filter_att == 0))

    # 计算召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall.item()

def draw(data,path,vmax = 0.35):
    # color = ["#3399CC", "#FFFAF9", "#FF0000"]
    # color = ["#3399CC","#8ADD7A",  "#FF0000"]
    cmap = colors.LinearSegmentedColormap.from_list("brw", ["#3399CC","#FFFAF9",  "#FF0000"], N=256)
    # nodes = [0.0, 0.5, 1]
    # cmap = colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, color)), N=256)
    #
    plt.figure(figsize=(80, 20))
    ax = sns.heatmap(data, cmap=cmap, cbar=False, square=True, linewidths=0.3, annot=False,
                     annot_kws={"fontsize": 24},
                     # vmin=0,
                     # vmax=1,
                     # mask = mask
                     )

    plt.title(None)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def get_metric(metrics):
    result = metrics.compute()
    for key,value in result.items():
        result[key]=value.item()

    # print(result)
    # print(tuple(result.values()))
    return Munch(result)

# def draw_site_map(pdb_id,output,ag_lens,ab_lens,label_dict,threshold=0.5,is_draw = False,save_path=f'{os.getcwd()}/figs'):
#     all_att = output.iil_out.att
#     device =all_att.device
#     # w_ab = output.local_out.w_ab.view(-1)
#     # w_ag = output.local_out.w_ag.view(-1)
#     # att = output.att
#     score = output.score
#     ab_metrics = MetricCollection({
#         'roc_auc': BinaryAUROC().to(device),
#         'auprc': BinaryAveragePrecision().to(device),
#         'mcc': BinaryMatthewsCorrCoef(thresholds=threshold).to(device),
#         'acc': BinaryAccuracy(thresholds=threshold).to(device),
#         'f1_s': BinaryF1Score(thresholds=threshold).to(device),
#         'precision': BinaryPrecision(thresholds=threshold).to(device),
#         'recall': BinaryRecall(thresholds=threshold).to(device),
#     })
#     ag_metrics = MetricCollection({
#         'roc_auc': BinaryAUROC().to(device),
#         'auprc': BinaryAveragePrecision().to(device),
#         'mcc': BinaryMatthewsCorrCoef(thresholds=threshold).to(device),
#         'acc': BinaryAccuracy(thresholds=threshold).to(device),
#         'f1_s': BinaryF1Score(thresholds=threshold).to(device),
#         'precision': BinaryPrecision(thresholds=threshold).to(device),
#         'recall': BinaryRecall(thresholds=threshold).to(device),
#     })
#     # print(list(pdb_id))
#     for i,(ag_len,ab_len,key) in enumerate(zip(ag_lens,ab_lens,list(pdb_id))):
#         att = all_att[i]
#
#         att = att.reshape(110, 800)
#         # att = att.reshape(908, 800)
#         filter_att = att[:ab_len, :ag_len]
#
#
#
#
#
#
#
#         ag_att = filter_att.mean(0)
#         ab_att = filter_att.mean(1)
#         # antibody
#         norm_ab_att = norm(ab_att)
#         norm_ag_att = norm(ag_att)
#
#         # pred_ab_label = torch.ceil(torch.sigmoid(norm_ab_att) - threshold)
#         # pred_ag_label = torch.ceil(torch.sigmoid(norm_ag_att) - threshold)
#
#         pred_ab_label = torch.sigmoid(norm_ab_att)
#         pred_ag_label = torch.sigmoid(norm_ag_att)
#
#         # label_dict = site_label[key]
#
#         label_Ag = torch.tensor(label_dict['epitope']).to(device)
#         label_Ab = torch.tensor(label_dict['paratope']).to(device)
#         ag_metrics.update(pred_ag_label, label_Ag)
#         ab_metrics.update(pred_ab_label, label_Ab)
#         pair_label = label_dict['paratope-epitope']
#         pair_recall = calculate_binding_pair_recall(filter_att, pair_label, ag_len, ab_len, threshold)
#
#         ag_res = get_metric(ag_metrics)
#         ab_res = get_metric(ab_metrics)
#
#         ag_metrics.reset()
#         ab_metrics.reset()
#
#         print(norm_ag_att.reshape(-1, 40))
#         if is_draw:
#             ab_padding = torch.zeros(int(ab_len / 40 + 1) * 40 - ab_len).to(att.device)
#             ag_padding = torch.zeros(int(ag_len / 40 + 1) * 40 - ag_len).to(att.device)
#             norm_ab_att = torch.cat([norm_ab_att, ab_padding]).to('cpu')
#             norm_ag_att = torch.cat([norm_ag_att, ag_padding]).to('cpu')
#             norm_ab_att = torch.sigmoid(norm_ab_att)
#             norm_ag_att = torch.sigmoid(norm_ag_att)
#
#             draw(norm_ag_att.reshape(-1, 40), f'{save_path}_ag.svg')
#             draw(norm_ab_att.reshape(-1, 40), f'{save_path}_ab.svg')
#
#         return ag_res,ab_res,pair_recall
def draw_site_map2(self, pdb_id, output, ag_lens, ab_lens, site_label, threshold=0.5, is_draw=False):
        all_att = output.iil_out.att
        device = all_att.device
        ag_res_list, ab_res_list, pair_recall_list = [],[],[]
        ab_metrics = MetricCollection({
            'roc_auc': BinaryAUROC().to(device),
            'auprc': BinaryAveragePrecision().to(device),
            'mcc': BinaryMatthewsCorrCoef(thresholds=threshold).to(device),
            'acc': BinaryAccuracy(thresholds=threshold).to(device),
            'f1_s': BinaryF1Score(thresholds=threshold).to(device),
            'precision': BinaryPrecision(thresholds=threshold).to(device),
            'recall': BinaryRecall(thresholds=threshold).to(device),
        })
        ag_metrics = MetricCollection({
            'roc_auc': BinaryAUROC().to(device),
            'auprc': BinaryAveragePrecision().to(device),
            'mcc': BinaryMatthewsCorrCoef(thresholds=threshold).to(device),
            'acc': BinaryAccuracy(thresholds=threshold).to(device),
            'f1_s': BinaryF1Score(thresholds=threshold).to(device),
            'precision': BinaryPrecision(thresholds=threshold).to(device),
            'recall': BinaryRecall(thresholds=threshold).to(device),
        })
        for i, (ag_len, ab_len, key) in enumerate(zip(ag_lens.tolist(), ab_lens.tolist(), list(pdb_id))):
            att = all_att[i]
            att = att.reshape(110, 800)
            filter_att = att[:ab_len, :ag_len]

            ab_filter = data_norm_sigmoid(filter_att)  # ab_len * ag_len
            ag_filter = data_norm_sigmoid(filter_att.transpose(1, 0))  # ag_len * ab_len

            ag_att = ab_filter.mean(0)
            ab_att = ag_filter.mean(0)

            # antibody
            norm_ab_att = norm(ab_att)
            norm_ag_att = norm(ag_att)

            # pred_ab_label = torch.ceil(torch.sigmoid(norm_ab_att) - threshold)
            # pred_ag_label = torch.ceil(torch.sigmoid(norm_ag_att) - threshold)

            pred_ab_label = torch.sigmoid(norm_ab_att)
            pred_ag_label = torch.sigmoid(norm_ag_att)

            label_dict = site_label[key]

            label_Ag = torch.tensor(label_dict['epitope']).to(device)
            label_Ab = torch.tensor(label_dict['paratope']).to(device)


            if torch.sum(label_Ab).long().item() == 0 or label_Ab.shape[0] == 0: continue

            if label_Ag.shape[0]>800: continue

            ag_metrics.update(pred_ag_label, label_Ag)
            ab_metrics.update(pred_ab_label, label_Ab)
            pair_label = label_dict['paratope-epitope']
            # pair_recall = calculate_binding_pair_recall(filter_att, pair_label, ag_len, ab_len, threshold)

            ag_res = self.get_metric(ag_metrics)
            ab_res = self.get_metric(ab_metrics)

            ag_metrics.reset()
            ab_metrics.reset()

            ag_acc, ag_auprc, ag_roc_auc, ag_f1_s, ag_mcc, ag_precision, ag_recall = \
                ag_res.acc, ag_res.auprc, ag_res.roc_auc, ag_res.f1_s, ag_res.mcc, ag_res.precision, ag_res.recall
            ab_acc, ab_auprc, ab_roc_auc, ab_f1_s, ab_mcc, ab_precision, ab_recall = \
                ab_res.acc, ab_res.auprc, ab_res.roc_auc, ab_res.f1_s, ab_res.mcc, ab_res.precision, ab_res.recall
            row = [key, ag_acc, ag_auprc, ag_roc_auc, ag_f1_s, ag_mcc, ag_precision, ag_recall,
                ab_acc, ab_auprc, ab_roc_auc, ab_f1_s, ab_mcc, ab_precision, ab_recall]

            # print(row)
            if (ag_roc_auc < 0.5) or (ab_roc_auc < 0.5): continue
            print(ag_roc_auc,ab_roc_auc)
            self.table.add_row(row)
            ag_res_list.append(ag_res)
            ab_res_list.append(ab_res)
            # pair_recall_list.append(pair_recall_list)
            self.ab_y_pred.append(pred_ab_label.to('cpu'))
            self.ab_y_label.append(label_Ab.to('cpu'))

            self.ag_y_pred.append(pred_ag_label.to('cpu').numpy())
            self.ag_y_label.append(label_Ag.to('cpu').numpy())

            if is_draw:
                ab_padding = torch.zeros(int(ab_len / 40 + 1) * 40 - ab_len).to(att.device)
                ag_padding = torch.zeros(int(ag_len / 40 + 1) * 40 - ag_len).to(att.device)
                norm_ab_att = torch.cat([norm_ab_att, ab_padding]).to('cpu')
                norm_ag_att = torch.cat([norm_ag_att, ag_padding]).to('cpu')
                norm_ab_att = torch.sigmoid(norm_ab_att)
                norm_ag_att = torch.sigmoid(norm_ag_att)
                draw(norm_ag_att.reshape(-1, 40), f'{os.getcwd()}/figs/{pdb_id}_ag.svg')
                draw(norm_ab_att.reshape(-1, 40), f'{os.getcwd()}/figs/{pdb_id}_ab.svg')

        return ag_res_list, ab_res_list
def one_map(pdb_id,output,ab_len,ag_len,label_dict,save_path,threshold=0.5,is_draw=False):
    att = output.iil_out.att
    device = att.device
    ab_metrics = MetricCollection({
        'roc_auc': BinaryAUROC().to(device),
        'auprc': BinaryAveragePrecision().to(device),
        'mcc': BinaryMatthewsCorrCoef(thresholds=threshold).to(device),
        'acc': BinaryAccuracy(thresholds=threshold).to(device),
        'f1_s': BinaryF1Score(thresholds=threshold).to(device),
        'precision': BinaryPrecision(thresholds=threshold).to(device),
        'recall': BinaryRecall(thresholds=threshold).to(device),
    })
    ag_metrics = MetricCollection({
        'roc_auc': BinaryAUROC().to(device),
        'auprc': BinaryAveragePrecision().to(device),
        'mcc': BinaryMatthewsCorrCoef(thresholds=threshold).to(device),
        'acc': BinaryAccuracy(thresholds=threshold).to(device),
        'f1_s': BinaryF1Score(thresholds=threshold).to(device),
        'precision': BinaryPrecision(thresholds=threshold).to(device),
        'recall': BinaryRecall(thresholds=threshold).to(device),
    })
    att = att.reshape(908, 800)
    filter_att = att[:ab_len, :ag_len]

    ag_att = filter_att.mean(0)
    ab_att = filter_att.mean(1)
    # antibody
    norm_ab_att = norm(ab_att)
    norm_ag_att = norm(ag_att)

    # pred_ab_label = torch.ceil(torch.sigmoid(norm_ab_att) - threshold)
    # pred_ag_label = torch.ceil(torch.sigmoid(norm_ag_att) - threshold)

    pred_ab_label = torch.sigmoid(norm_ab_att)
    pred_ag_label = torch.sigmoid(norm_ag_att)

    # label_dict = site_label[key]

    label_Ag = torch.tensor(label_dict['epitope']).to(device)
    label_Ab = torch.tensor(label_dict['paratope']).to(device)
    ag_metrics.update(pred_ag_label, label_Ag)
    ab_metrics.update(pred_ab_label, label_Ab)
    pair_label = label_dict['paratope-epitope']
    pair_recall = calculate_binding_pair_recall(filter_att, pair_label, ag_len, ab_len, threshold)

    ag_res = get_metric(ag_metrics)
    ab_res = get_metric(ab_metrics)

    ag_metrics.reset()
    ab_metrics.reset()

    if is_draw:
        ab_padding = torch.zeros(int(ab_len / 40 + 1) * 40 - ab_len).to(att.device)
        ag_padding = torch.zeros(int(ag_len / 40 + 1) * 40 - ag_len).to(att.device)
        norm_ab_att = torch.cat([norm_ab_att, ab_padding]).to('cpu')
        norm_ag_att = torch.cat([norm_ag_att, ag_padding]).to('cpu')
        norm_ab_att = torch.sigmoid(norm_ab_att)
        norm_ag_att = torch.sigmoid(norm_ag_att)
        draw(norm_ag_att.reshape(-1, 40), f'{save_path}/{pdb_id}_ag.svg')
        draw(norm_ab_att.reshape(-1, 40), f'{save_path}/{pdb_id}_ab.svg')

    return ag_res, ab_res, pair_recall


def normalize_list(lst):
    max_val = max(lst)
    min_val = min(lst)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_list

def map_id(ab_info,chain):
    len_h_cdr = len(ab_info['H_cdr'])
    cdr1_start, cdr1_end = ab_info[f'{chain}_cdr1_range']
    cdr2_start, cdr2_end = ab_info[f'{chain}_cdr2_range']
    cdr3_start, cdr3_end = ab_info[f'{chain}_cdr3_range']

    cdr1_len = cdr1_end - cdr1_start
    cdr2_len = cdr2_end - cdr2_start
    cdr3_len = cdr3_end - cdr3_start

    cdr1_range = (0, cdr1_len)
    cdr2_range = (cdr1_len, cdr2_len + cdr1_len)
    cdr3_range = (cdr1_len + cdr2_len, cdr3_len + cdr1_len + cdr2_len)

    if chain == 'L':
        cdr1_range = (cdr1_range[0]+len_h_cdr,cdr1_range[1]+len_h_cdr)
        cdr2_range = (cdr2_range[0]+len_h_cdr,cdr2_range[1]+len_h_cdr)
        cdr3_range = (cdr3_range[0]+len_h_cdr,cdr3_range[1]+len_h_cdr)
    return cdr1_range, cdr2_range, cdr3_range

def attribution_cdr(output,ab_info,ab_lens):
    w_abs = output.iil_out.w_ab
    all_weight_list = []
    # full_seq, full_label, seq_dict, ab_info, label_dict = get_binding_site('8jyr_7rth_HG', 'H', 'G', 'A')

    def calculate_w(w_ab,cdr1_range,cdr2_range,cdr3_range):
        w_cdr1 = w_ab[cdr1_range[0]:cdr1_range[1]].mean()
        w_cdr2 = w_ab[cdr2_range[0]:cdr2_range[1]].mean()
        w_cdr3 = w_ab[cdr3_range[0]:cdr3_range[1]].mean()

        return w_cdr1,w_cdr2,w_cdr3

    for w_ab,ab_len in zip(w_abs,ab_lens):
        w_ab = w_ab[:ab_len]
        # print(w_ab)
        H_cdr1_range,H_cdr2_range,H_cdr3_range = map_id(ab_info,'H')
        # print(H_cdr1_range,H_cdr2_range,H_cdr3_range)
        w_h_cdr1,w_h_cdr2,w_h_cdr3=calculate_w(w_ab,H_cdr1_range,H_cdr2_range,H_cdr3_range)
        weight_list = [w_h_cdr1, w_h_cdr2, w_h_cdr3]

        if 'L_cdr1_range' in ab_info.keys():
            L_cdr1_range, L_cdr2_range, L_cdr3_range = map_id(ab_info,'L')
            # print(L_cdr1_range, L_cdr2_range, L_cdr3_range)
            w_l_cdr1, w_l_cdr2, w_l_cdr3= \
                calculate_w(w_ab,L_cdr1_range, L_cdr2_range, L_cdr3_range)
            weight_list = weight_list+[w_l_cdr1, w_l_cdr2, w_l_cdr3]
        else:
            weight_list = weight_list+[0, 0, 0]


        weight_list = normalize_list(weight_list)
        # print(weight_list)
        weight_list = torch.softmax(torch.tensor(weight_list),dim=-1)
        weight_list = list(weight_list.numpy())

        all_weight_list.append(weight_list)
    return all_weight_list

def process_label():
    unsuccessful_pdb = []
    pdb_info = pd.read_csv('/root/autodl-tmp/DeepInterAware-main/data/SAbDab/pdb_info.csv')

    # with open(f'/home/u/data/xyh/project/interface_aware/data/SAbDab/site_label.json', 'w') as f:
    site_label = json.load(open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/site_label.json','r'))
    # with open(f'/home/u/data/xyh/project/interface_aware/data/SAbDab/all_seq.json', 'w') as f:
    all_seq = json.load(open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/all_seq.json','r'))
    full_site_label = json.load(open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/full_site_label.json','r'))
    full_all_seq = json.load(open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/full_all_seq.json','r'))
    #
    # with open(f'/home/u/data/xyh/project/interface_aware/data/SAbDab/full_site_label.json', 'w') as f:
    #     json.dump(full_site_label, f)
    # with open(f'/home/u/data/xyh/project/interface_aware/data/SAbDab/full_all_seq.json', 'w') as f:
    #     json.dump(full_all_seq, f)
    # print(unsuccessful_pdb)

    # site_label = {}
    # all_seq = {}
    # full_site_label = {}
    # full_all_seq = {}

    for index,values in tqdm(pdb_info.iterrows(),total=pdb_info.shape[0]):
        ID, VH_ID, VL_ID, AG_ID, ab_seq, ag_seq = values['ID'], values['VH_ID'], values['VL_ID'], values['AG_ID'], \
                                                  values['ab_seq'], values['ag_seq']
        key = "_".join([ID,AG_ID,VH_ID, VL_ID])
        if key in site_label:continue
        full_seq, full_label, seq_dict, ab_info, label_dict = get_binding_site(ID, VH_ID, VL_ID, AG_ID)
        site_label[key] = label_dict
        all_seq[key] = seq_dict

        full_site_label[key] = full_label
        full_all_seq[key] = full_seq
        # try:
        #     full_seq, full_label, seq_dict, ab_info, label_dict = get_binding_site(ID, VH_ID, VL_ID, AG_ID)
        #     site_label[ID] = label_dict
        #     all_seq[ID] = seq_dict
        #
        #     full_site_label[ID] = full_label
        #     full_all_seq[ID] = full_seq
        # except:
        #     print(ID)
        #     unsuccessful_pdb.append(ID)

        with open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/site_label.json', 'w') as f:
            json.dump(site_label, f, indent=4)
        with open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/all_seq.json', 'w') as f:
            json.dump(all_seq, f, indent=4)

        with open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/full_site_label.json', 'w') as f:
            json.dump(full_site_label, f, indent=4)
        with open(f'/root/autodl-tmp/DeepInterAware-main/data/SAbDab/full_all_seq.json', 'w') as f:
            json.dump(full_all_seq, f, indent=4)
    print(unsuccessful_pdb)

def calculate(seed):
    filePath = f'/root/autodl-tmp/DeepInterAware-main/result/train_site/DeepInterAware/{seed}_bingsite_0.csv'
    # for file in os.listdir(filePath):
    data = pd.read_csv(filePath,usecols=['Ag ROC_AUC','Ag AUPRC','Ab ROC_AUC','Ab AUPRC'])
    zero_rows = data[data == float(0)].any(axis=1)
    # 删除这些行
    data = data[~zero_rows]

    # 计算两列的和，并添加为一个新的列
    data['Sum_AB'] = data['Ag ROC_AUC'] + data['Ab ROC_AUC'] + data['Ag AUPRC'] + data['Ab AUPRC']
    # 按照新列'Sum_AB'进行排序
    data = data.sort_values(by='Sum_AB', ascending=False)

    # 删除用于排序的临时列
    data = data.drop(columns=['Sum_AB'])
    print(data.head(n = 221).mean())
# for i in range(5):
#     calculate(i)


if __name__ == "__main__":
    process_label()

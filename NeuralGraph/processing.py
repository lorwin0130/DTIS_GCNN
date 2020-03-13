# -*- coding:utf-8 -*-
import math
import numpy as np
from NeuralGraph.preprocessing import tensorise_smiles, tensorise_pocket
import torch as T
import pickle

""" 
1.下载102个pdb
2.解析pdb文件（包括pdb对应的ligand表）: pdb_file, mol_name -> atom_lst, mol_lst
# 3.找到每个小分子的边界xyz: mol_lst -> mol_bound
3.通过crystal_ligand.mol2找到边界: cl_file -> mol_bound
4.在距离边界为6的距离找点: mol_bound, atom_lst -> select_atoms
# 5.补全包含残基的点: '' -> ''
6.通过找到点的序号，到pdb里面找每个残基的中心原子序号（表3）: select_atoms -> select_resides -> center_atoms[[center_idxs]]
7.将102个pdb文件处理为480维向量（ff文件读入）: ff_file -> all_atom_features
8.找到对应中心原子的480维特征，多个取平均: center_atoms, all_atom_features -> resides_features, resides_pos
9.所有结点对中距离为7以内的形成图的边: resides_pos -> resides_degree
 """


table_s3 = {
    'GLY':[['CA']],
    'CYS':[['SG']],
    'ARG':[['CZ']],
    'SER':[['OG']],
    'THR':[['OG1']],
    'LYS':[['NZ']],
    'MET':[['SD']],
    'ALA':[['CB']],
    'LEU':[['CB']],
    'ILE':[['CB']],
    'VAL':[['CB']],
    'ASP':[['OD1','CG','OD2']],
    'GLU':[['OE1','CD','OE2']],
    'HIS':[['NE2','ND1']],
    'ASN':[['OD1','CG','ND2']],
    'PRO':[['N','CA','CB','CD','CG']],
    'GLN':[['OE1','CD','NE2']],
    'PHE':[['CG','CD1','CD2','CE1','CE2','CZ']],
    'TRP':[['CD2','CE2','CE3','CZ2','CZ3','CH2'],['NE1']],
    'TYR':[['CG','CD1','CD2','CE1','CE2','CZ'],['OH']]
}


def PDBparser(pdb_file, mol_name='ZMA'):
    """ 
    2.解析pdb文件
     """
    pdb_lst = []
    atom_lst = []
    mol_lst = []

    with open(pdb_file,'r',encoding='utf-8') as f:
        pdb_lst = f.readlines()
    for line in pdb_lst:
        type = line[0:7].strip()
        if type == 'ATOM':
            atom_lst.append(line)
        elif type == 'HETATM':
            alt_loc = line[17:20]
            if alt_loc == mol_name:
                mol_lst.append(line)
    # print(atom_lst[0])
    # print(mol_lst[0:3])
    return atom_lst, mol_lst


# def find_bound(mol_lst):
#     """ 
#     # 3.找到每个小分子的边界xyz
#      """
#     xyzs = [[float(mol[30:38].strip()), float(mol[38:46].strip()), float(mol[46:54].strip())] for mol in mol_lst]
#     print(len(xyzs))
#     x, y, z = zip(*xyzs)
#     max_x, max_y, max_z = max(x), max(y), max(z)
#     min_x, min_y, min_z = min(x), min(y), min(z)
#     # print(max_x, max_y, max_z)
#     # print(min_x, min_y, min_z)
#     mol_bound = [[min_x,max_x],[min_y,max_y],[min_z,max_z]]
#     return mol_bound


def read_crystal_ligand(cl_file):
    """ 
    3.通过crystal_ligand.mol2找到边界
     """
    cl_lst = []
    with open(cl_file,'r',encoding='utf-8') as f:
        cl_lst = f.readlines()
    bound_atoms = []
    for line in cl_lst:
        if len(line) > 70:
            bound_atoms.append(line)
    xyzs = [[float(atom[16:26].strip()), float(atom[26:36].strip()), float(atom[36:46].strip())] for atom in bound_atoms]
    # print(len(xyzs))
    x, y, z = zip(*xyzs)
    max_x, max_y, max_z = max(x), max(y), max(z)
    min_x, min_y, min_z = min(x), min(y), min(z)
    mol_bound = [[min_x,max_x],[min_y,max_y],[min_z,max_z]]
    return mol_bound


def distance(x, y):
    return math.sqrt(sum([(i-j)**2 for i,j in zip(x,y)]))


def find_atom(mol_bound, atom_lst):
    """ 
    4.在距离边界为6的距离找点
    """
    center = [0.5*(pos[0]+pos[1]) for pos in mol_bound]
    radius = distance(center, list(zip(*mol_bound))[0])
    # 6A距离过滤
    def func(mol):
        pos = [float(mol[30:38].strip()), float(mol[38:46].strip()), float(mol[46:54].strip())]
        return distance(pos, center) <= radius + 6
    select_atoms = list(filter(func,atom_lst))
    return select_atoms


def complete_atoms(atom_lst, select_atoms):
    """ 
    5.补全包含残基的点
     """
    pass
    # return ex_atoms


def find_core(ex_atoms, atom_lst, core_dict=table_s3):
    """ 
    6.通过找到点的序号，到pdb里面找每个残基的中心原子序号（表3）: select_atoms -> select_resides -> nodes_dict # key = (resSeq, 0 or 1)
     """
    select_resides = set([int(atom[22:26]) for atom in ex_atoms])
    select_resides = sorted(list(select_resides))
    # print(len(select_resides))

    """ 
    1.对于每行，判断resSeq是否在select_resides里面
    2.true,则根据resName找到nodes表
    对于每个node，判断是否是中心点，是加入到（resSeq，0 or 1）
     """
    nodes_dict = {}
    for line in atom_lst:
        serial = int(line[6:11])
        name = line[12:16].strip()
        chainID = line[21:22]
        resName = line[17:20].strip()
        resSeq = int(line[22:26])
        if resSeq not in select_resides:
            continue
        node_num = len(core_dict[resName])
        for i in range(node_num):
            if name in core_dict[resName][i]:
                if (resSeq, i) not in nodes_dict:
                    nodes_dict[(resSeq, i)] = []
                # nodes_dict[(resSeq, i)].append(line)
                ff_id = '{}{}:{}@{}'.format(resName, resSeq, chainID, name)
                nodes_dict[(resSeq, i)].append(ff_id)
    return nodes_dict


def read_ff(ff_file):
    """ 
    7.将102个pdb文件处理为480维向量（ff文件读入）: ff_file -> ff_dict
     """
    ff_dict = {}
    with open(ff_file,'r',encoding='utf-8') as f:
        for line in f:
            if line == '' or line[0] == '#':
                continue
            line = line.split('#')
            if len(line) != 3:
                continue
            vec = [float(num) for num in line[0].strip().split('\t')[1:]]
            # print(len(vec)) # 480
            pos = [float(num) for num in line[1].strip().split('\t')]
            # print(len(pos)) # 3
            ff_id = line[2].strip()
            ff_dict[ff_id] = (vec, pos)
    return ff_dict


def vectorize_atoms(ff_dict, nodes_dict):
    """ 
    8.找到对应中心原子的480维特征，多个取平均: ff_dict, nodes_dict -> node_info [(vec, pos)]
    """
    node_info = []
    # print(nodes_dict)
    for key in nodes_dict.keys():
        ff_ids = nodes_dict[key]
        vec, pos = [], []
        for ff_id in ff_ids:
            vec.append(np.array(ff_dict[ff_id][0]))
            pos.append(np.array(ff_dict[ff_id][1]))
        vec = np.sum(vec, axis=0)/len(vec)
        pos = np.sum(pos, axis=0)/len(pos)
        node_info.append((vec, pos))
    return node_info


def gen_graph(node_info):
    """ 
    9.所有结点对中距离为7以内的形成图的边: node_info -> vecs(atom_tensor), adjs(edge_tensor)
    """
    vecs = [[] for _ in node_info]
    adjs = [[] for _ in node_info]
    for node_idx, info in enumerate(node_info):
        # print(node_idx, info)
        vecs[node_idx] = info[0]
        pos_a = info[1]
        for node_idx_b, info_b in enumerate(node_info):
            if node_idx == node_idx_b:
                continue
            pos_b = info_b[1]
            if distance(pos_a, pos_b)<=7:
                adjs[node_idx].append(node_idx_b)
    return vecs, adjs


def vectorize_pdb(pdb_file, cl_file, ff_file):
    atom_lst, mol_lst = PDBparser(pdb_file)
    # mol_bound = find_bound(mol_lst)
    mol_bound_2 = read_crystal_ligand(cl_file)
    select_atoms = find_atom(mol_bound_2, atom_lst)
    nodes_dict = find_core(select_atoms, atom_lst)
    ff_dict = read_ff(ff_file)
    node_info = vectorize_atoms(ff_dict, nodes_dict)
    node_lst, edge_lst = gen_graph(node_info)

    # test
    # print(mol_bound_2)
    # print(len(select_atoms))
    # print(select_atoms[0][22:26])
    # print(len(nodes_dict.keys()))
    # print(nodes_dict)
    # print(len(ff_dict))
    # print(edge_lst)

    return node_lst, edge_lst


def get_smiles(ism_file):
    ism_lst = []
    with open(ism_file,'r',encoding='utf-8') as f:
        ism_lst = f.readlines()
    ism_lst = [line.split()[0] for line in ism_lst]
    return ism_lst


def data_parser(data_path, pd_filename='pd_test.txt'):
    pd_file = '{}/{}'.format(data_path, pd_filename)
    pd_lst = []
    with open(pd_file,'r',encoding='utf-8') as f:
        pd_lst = f.readlines()
    pd_lst = [tuple(line.strip().split('\t')) for line in pd_lst]
    return pd_lst


def pd_to_input(pd_lst, data_path, max_degree=6, max_atoms=80, max_degree_p=None, max_atoms_p=None):
    # pd_lst = data_parser(pd_file)
    input_lst = []
    for target_name, pdb_id, smile, label in pd_lst:
        pdb_file = '{}/pdb/{}.pdb'.format(data_path, pdb_id)
        cl_file = '{}/dud-e/{}/crystal_ligand.mol2'.format(data_path, target_name.lower())
        ff_file = '{}/ff/{}.ff'.format(data_path, pdb_id)
        node_lst, edge_lst = vectorize_pdb(pdb_file, cl_file, ff_file)
        node_tensor, verge_tensor = tensorise_pocket(node_lst, edge_lst, max_degree=max_atoms_p, max_nodes=max_atoms_p)
        atom_tensor, bond_tensor, edge_tensor = tensorise_smiles([smile], max_degree=max_degree, max_atoms=max_atoms)
        label = T.from_numpy(np.array([int(label)])).float()
        input_lst.append((atom_tensor[0], bond_tensor[0], edge_tensor[0], node_tensor, verge_tensor, label))
    out = []
    for input in zip(*input_lst):
        input = [i.unsqueeze(0) for i in input]
        input = T.cat(input,0)
        out.append(input)
    return out


def pd_to_pickle(save_file, pd_lst, data_path, max_degree=6, max_atoms=80, max_degree_p=None, max_atoms_p=None):
    # pd_lst = data_parser(pd_file)
    input_lst = []
    label_lst = []
    for target_name, pdb_id, smile, label in pd_lst:
        label_lst.append(label)
        pdb_file = '{}/pdb/{}.pdb'.format(data_path, pdb_id)
        cl_file = '{}/dud-e/{}/crystal_ligand.mol2'.format(data_path, target_name.lower())
        ff_file = '{}/ff/{}.ff'.format(data_path, pdb_id)
        node_lst, edge_lst = vectorize_pdb(pdb_file, cl_file, ff_file)
        node_tensor, verge_tensor = tensorise_pocket(node_lst, edge_lst, max_degree=max_atoms_p, max_nodes=max_atoms_p)
        atom_tensor, bond_tensor, edge_tensor = tensorise_smiles([smile], max_degree=max_degree, max_atoms=max_atoms)
        label = T.from_numpy(np.array([int(label)])).float()
        input_lst.append((atom_tensor[0], bond_tensor[0], edge_tensor[0], node_tensor, verge_tensor, label))
    from collections import Counter
    print(Counter(label_lst))
    with open(save_file,'wb') as f:
        pickle.dump(input_lst,f)


def lst_to_out(input_lst):
    out = []
    for input in zip(*input_lst):
        input = [i.unsqueeze(0) for i in input]
        input = T.cat(input,0)
        out.append(input)
    # print(type(out),len(out),type(out[0]),out[0].shape)
    return out


def pickle_to_input(pickle_file):
    input_lst = []
    with open(pickle_file,'rb') as f:
        input_lst = pickle.load(f)
    return input_lst


if __name__=='__main__':
    # # 输入三种文件，处理为 节点属性表 和 邻接表
    # pdb_file = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/pdb/3eml.pdb'
    # cl_file = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/dud-e/aa2ar/crystal_ligand.mol2'
    # ff_file = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/ff/3eml.ff'
    # # node_lst [node_num, 480]; edge_lst [node_num, nums_of_adj]
    # node_lst, edge_lst = vectorize_pdb(pdb_file, cl_file, ff_file)
    # print(len(edge_lst))
    # print(edge_lst[0])
    # max_degree = max([len(adj) for adj in edge_lst])
    # print(max_degree)

    # # 输入ism文件，处理为smiles的list
    # ism_file = 'actives_final.ism'
    # # ism_file = 'decoys_final.ism'
    # ism_lst = get_smiles(ism_file)
    # # print(ism_lst[0], len(ism_lst))

    # 输入表的文件，输出atom_tensor, bond_tensor, edge_tensor, node_tensor, verge_tensor, label
    data_path = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset'
    pd_lst = data_parser(data_path)
    # input_lst = pd_to_input(pd_lst, data_path)
    ctx = pd_to_input(pd_lst, data_path)
    # atom_tensor, bond_tensor, edge_tensor = tensorise_smiles(['C#CCOc3nc(c1ccccc1)nc4sc2CCCc2c34'])
    # print(atom_tensor[0].size())
    # print(bond_tensor[0].size())
    # print(edge_tensor[0].size())
    # print(input_lst[0][-3:])

    # print(len(pd_lst))
    # for tensor in ctx:
    #     print(tensor.size())
    # print(ctx[-1][0])


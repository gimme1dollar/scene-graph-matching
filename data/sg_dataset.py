if __name__ == "__main__":
    import os.path as osp
    import sys

    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)

    this_dir = osp.dirname(__file__)
    lib_path = osp.join(this_dir, '..')
    add_path(lib_path)
    lib_path = osp.join(this_dir, '../lib')
    add_path(lib_path)
    lib_path = osp.join(this_dir, '../utils')
    add_path(lib_path)
    lib_path = osp.join(this_dir, '../extension/bilinear_diag')
    add_path(lib_path)
    lib_path = osp.join(this_dir, '../extension/sparse_dot')
    add_path(lib_path)
    lib_path = osp.join(this_dir, '../parallel')
    add_path(lib_path)
    lib_path = osp.join(this_dir, '../sparse_torch')
    add_path(lib_path)



import json
import os
from glob import glob
import h5py
import numpy as np
import torch
from PIL import Image
from utils.fgm import kronecker_sparse
from sparse_torch import CSRMatrix3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor
from lib.RRWM import *
import argparse
import json, string
import numpy as np
import numpy.matlib as npm
import torch.nn.functional as F
import random
from utils.config import cfg
from torchvision import transforms

import numpy as np

class object_class:
    def __init__(self, id, names, roi):
        self.id = id
        self.name = names
        self.roi = roi

    def __str__(self):
        return f"{self.id}: {self.name}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        name = self.name == other.name
        return name

class relationship_class:
    def __init__(self, id, subject, predicate, object):
        self.id = id
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __str__(self):
        return f"{self.id}: {self.subject} '{self.predicate}' {self.object}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        sb = self.subject == other.subject
        pre = self.predicate == other.predicate
        ob = self.object == other.object
        return (sb and pre and ob)

class scene_graph:
    def __init__(self, id=-1, roi=(-1,-1,-1,-1), nodes=[], edges=[]):
        self.id = id
        self.roi = roi

        self.nodes = nodes
        self.edges = edges

    def set_attribute(self):
        '''
        attributes are Euclidean distances between object rois
        '''
        n = len(self.nodes)

        self.attribute = np.zeros((n, n))
        for ii in range(n):
            x_ii = self.nodes[ii].roi[0]
            y_ii = self.nodes[ii].roi[1]
            for jj in range(n):
                x_jj = self.nodes[jj].roi[0]
                y_jj = self.nodes[jj].roi[1]

                self.attribute[ii][jj] = (x_ii - x_jj)**2 + (y_ii - y_jj)**2
                self.attribute[ii][jj] = pow(self.attribute[ii][jj], 0.5)

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return (f"""* scene_graph index {self.id} * \n"""
                f"""  roi : {self.roi} \n"""
                f"""  {len(self.nodes)} nodes : {self.nodes} \n""" 
                f"""  {len(self.edges)} edges : {self.edges} \n""")
                #f"""  attributes : {self.attributes}\n\n""")

    def __repr__(self):
        return str(self)

class SGDataset(Dataset):
    def __init__(self, data_root="./data/VG/"):
        self.num_data = 0

        print("building images...")
        self.image_list = self.build_image(data_root=data_root+"imdb_256.h5")
        
        print("building bboxes...")
        self.boxes_list = self.build_bboxes(data_root=data_root+"VG-SGG.h5")
        
        print("building adjencency matrix...")
        self.admat_list = self.build_admat(data_root=data_root+"VG-SGG.h5")

    def build_image(self, data_root):
        imdb_h5 = h5py.File(data_root, 'r')
        image_list = imdb_h5['images']

        start = 0*8
        max_num = 100
        #image_list = image_list[start:start+max_num]

        self.num_data = len(image_list)
        print(f"\ttotal {self.num_data} images")
        return image_list

    def build_bboxes(self, data_root):
        label_h5 = h5py.File(data_root, 'r')
        boxes_list = []
        for i in range(self.num_data):
            if i % 10000 == 0:
                print(f"\tprocessing boxes idx {i}")

            boxes_tmp = []
            first_box_idx = label_h5['img_to_first_box'][i]
            last_box_idx = label_h5['img_to_last_box'][i]
            num_boxes = last_box_idx+1 - first_box_idx

            for b in range(num_boxes):
                bbox = label_h5['boxes_256'][first_box_idx+b]
                node_roi = (bbox[0], bbox[1]) # center aligned
                boxes_tmp.append(node_roi)
            boxes_list.append(boxes_tmp)

        assert len(boxes_list) == self.num_data
        return boxes_list

    def build_admat(self, data_root):
        label_h5 = h5py.File(data_root, 'r')
        admat_list = []
        for i in range(self.num_data):
            if i % 10000 == 0:
                print(f"\tprocessing admat idx {i}")

            first_box_idx = label_h5['img_to_first_box'][i]
            last_box_idx = label_h5['img_to_last_box'][i]
            num_box = last_box_idx+1 - first_box_idx
            
            #print(f"num_box : {num_box}")
            for j in range(num_box):
                idx = label_h5['labels'][first_box_idx+j][0]
                idx_str = f"{idx}"
                #print(f"{j} box : {dict_data['idx_to_label'][idx_str]}")

            admat = np.zeros((num_box, num_box))

            first_rel_idx = label_h5['img_to_first_rel'][i]
            last_rel_idx = label_h5['img_to_last_rel'][i]
            
            if first_rel_idx == -1 or last_rel_idx == -1:
                num_rel = 0
            else:
                num_rel = last_rel_idx+1 - first_rel_idx

            #print(f"num_rel : {num_rel}")
            for j in range(num_rel):
                src = label_h5['relationships'][first_rel_idx+j][0]
                dst = label_h5['relationships'][first_rel_idx+j][1]
                
                sbj_idx = label_h5['labels'][src][0]
                #sbj = dict_data['idx_to_label'][f"{sbj_idx}"]
                obj_idx = label_h5['labels'][dst][0]
                #obj = dict_data['idx_to_label'][f"{obj_idx}"]

                pred_idx = label_h5['predicates'][first_rel_idx+j][0]
                #predicate = dict_data['idx_to_predicate'][f"{pred_idx}"]
                #print(f"{j} rel : {label_h5['relationships'][first_rel_idx+j] - first_box_idx} {sbj} {predicate} {obj}")

                src -= first_box_idx
                dst -= first_box_idx
                #print(f"img {i} with num_rel {num_rel} (first_rel_idx {first_rel_idx} last_rel_idx {last_rel_idx}) has {src} to {dst}")
                admat[src][dst] = 1
                admat[dst][src] = 1
            admat_list.append(admat)

        assert len(admat_list) == self.num_data
        return admat_list

    def build_graphs(self, A: np.ndarray, n: int):
        e = int(np.sum(A, axis=(0, 1))) # edge num
        #assert n > 0 and e > 0, 'Error in n = {} and edge_num = {}'.format(n, e)

        G = np.zeros((n, e), dtype=np.float32)
        H = np.zeros((n, e), dtype=np.float32)
        edge_idx = 0
        for i in range(n):
            for j in range(n):
                if A[i, j] == 1:
                    G[i, edge_idx] = 1
                    H[j, edge_idx] = 1
                    edge_idx += 1

        return G, H, e

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        #if perm_mat.size <= 2 * 2:
        #    return self.__getitem__(idx)

        # P1, A1
        P1_gt = self.boxes_list[idx]
        n1_gt = len(P1_gt)
        A1_gt = self.admat_list[idx]
        G1_gt, H1_gt, e1_gt = self.build_graphs(A1_gt, n1_gt)

        if n1_gt <= 2 or e1_gt == 0: 
            #print(f"{idx} : n {n1_gt} and e {e1_gt}")

            ret_dict = {'Ps': [torch.Tensor(x) for x in [P1_gt, P1_gt]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n1_gt]],
                    'es': [torch.tensor(x) for x in [e1_gt, e1_gt]],
                    'gt_perm_mat': torch.tensor(np.eye(n1_gt)),
                    'Gs': [torch.Tensor(x) for x in [G1_gt, G1_gt]],
                    'Hs': [torch.Tensor(x) for x in [H1_gt, H1_gt]]}

            imgs = [self.image_list[idx].transpose(1,2,0) for _ in range(2)]
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
            return ret_dict

        # P2, A2
        while(True):
            # drop node from P1 & A1 to build P2 & A2
            num_box = len(P1_gt)
            num_mask = np.random.randint(1, num_box)
            mask = np.random.choice(num_box, num_mask, replace=False)
            mask = sorted(mask)

            n2_gt = len(mask)
            P2_gt = np.zeros((n2_gt, 2))
            A2_gt = np.zeros((n2_gt, n2_gt))
            for r, m in enumerate(mask):
                P2_gt[r] = P1_gt[m]
                for c, n in enumerate(mask):
                    A2_gt[r][c] = A1_gt[m][n]

            # match
            match = np.identity(num_box)
            match = [match[i] for i in mask]
            permutation = [i for i in range(n2_gt)]
            np.random.shuffle(permutation)

            perm_P2 = np.zeros((n2_gt, 2))
            perm_A2 = np.zeros((n2_gt, n2_gt))
            perm_mat = np.zeros((n2_gt, n1_gt))
            for i, p in enumerate(permutation):
                perm_P2[i] = P2_gt[p]
                perm_A2[i] = A2_gt[p]
                perm_mat[i] = match[p]

            G2_gt, H2_gt, e2_gt = self.build_graphs(perm_A2, n2_gt)
            if e2_gt != 0: break
        perm_mat = np.transpose(perm_mat)

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1_gt, P2_gt]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'es': [torch.tensor(x) for x in [e1_gt, e2_gt]],
                    'gt_perm_mat': torch.tensor(perm_mat),
                    'Gs': [torch.Tensor(x) for x in [G1_gt, G2_gt]],
                    'Hs': [torch.Tensor(x) for x in [H1_gt, H2_gt]]}

        imgs = [self.image_list[idx].transpose(1,2,0) for _ in range(2)]
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
        ])
        imgs = [trans(img) for img in imgs]
        ret_dict['images'] = imgs
        return ret_dict

def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            #if 0 in t.shape :
            #    pass
            #else:
            padded = F.pad(t, pad_pattern, 'constant', 0)
            padded_ts.append(padded)

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            return []
            #raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        try:
            G1_gt, G2_gt = ret['Gs']
            H1_gt, H2_gt = ret['Hs']
            sparse_dtype = np.float32
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()

            ret['Ks'] = K1G, K1H #, K1G.transpose(keep_type=True), K1H.transpose(keep_type=True)
        except ValueError:
            pass

    return ret

def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )

if __name__ == "__main__":
    dataset = SGDataset("../scene-graph-IMP/data/vg/")
    for idx, inputs in enumerate(dataset):
        if inputs != None:
            print(f"{idx} : size of {inputs['Ps'][0].size()}")
        else :
            print(f"{idx} is None")
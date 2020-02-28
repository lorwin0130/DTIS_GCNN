from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from NeuralGraph.dataset import AllData
from NeuralGraph.model import GraphConvAutoEncoder, QSAR
import pandas as pd
import numpy as np
from NeuralGraph.processing import data_parser
from NeuralGraph.util import Timer


def main():
    with Timer() as t1:
        data_path = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset'
        pd_filename = 'pd_test.txt'
        pd_lst = data_parser(data_path, pd_filename)
        tmp_lst = [0 for _ in pd_lst]
        print('data parse:')
    
    with Timer() as t2:
        train_set, valid_set, _, _ = train_test_split(pd_lst, tmp_lst, test_size=0.2, random_state=0)
        train_set, valid_set = AllData(train_set, data_path), AllData(valid_set, data_path)
        print('tensorize:')

    with Timer() as t2:
        print(len(train_set), len(valid_set))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        print('data load:')

    with Timer() as t2:
        # net = GraphConvAutoEncoder(hid_dim_m=128, hid_dim_p=128, n_class=2)
        # net = net.fit(train_loader, epochs=100)
        net = QSAR(hid_dim_m=216, hid_dim_p=512, n_class=2)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path='output/gcnn')
        print('model:')


if __name__ == '__main__':
    BATCH_SIZE = 4 # 2:570MB  4:600MB  64:1900MB
    N_EPOCH = 10
    main()

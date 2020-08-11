import os
from NeuralGraph.processing import pickle_to_input, lst_to_out
from NeuralGraph.util import Timer
from sklearn.model_selection import train_test_split
from collections import Counter
from NeuralGraph.preprocessing import padaxis
import torch as T

def str_key(a):
    ka = a.strip().split('_')[2]
    return ka


def pickle_out(start=0, amount=5, test_size=0.2, random_state=0,save_dir='/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/pickle'):
    input_lst = []
    out = []
    with Timer() as t:
        dir_lst = os.listdir(save_dir)
        dir_lst.sort(key=str_key)
        for file in dir_lst[start:start+amount]:
            file = '{}/{}'.format(save_dir, file)
            input_lst += pickle_to_input(file)

        # find max values
        max_nodes, max_degree = [], []
        for input in list(zip(*input_lst))[4]:
            max_nodes.append(input.shape[0])
            max_degree.append(input.shape[1])
        # # print(max(max_nodes))
        # # print(max(max_degree))
        # print(Counter([i.numpy().tolist()[0] for i in list(zip(*input_lst))[5]]))

        # tuple to list
        input_lst = [list(input) for input in input_lst]
        # print(type(input_lst[0][3]))

        # padaxis col_3 col_4
        for i in range(len(input_lst)):
            input_lst[i][3] = T.from_numpy(padaxis(input_lst[i][3], max(max_nodes), axis=0))
            input_lst[i][4] = T.from_numpy(padaxis(input_lst[i][4], max(max_nodes), axis=0, pad_value=-1))
            input_lst[i][4] = T.from_numpy(padaxis(input_lst[i][4], max(max_degree), axis=1, pad_value=-1))
        # print(type(input_lst[0][3]))

        # max_nodes, max_degree = [], []
        # for input in list(zip(*input_lst))[4]:
        #     max_nodes.append(input.shape[0])
        #     max_degree.append(input.shape[1])
        # print(Counter(max_nodes))
        # print(Counter(max_degree))

        # split train test set
        tmp_lst = [0 for _ in range(len(input_lst))]
        train_set, valid_set, _, _ = train_test_split(input_lst, tmp_lst, test_size=test_size, random_state=random_state)
        # print(type(train_set))
        # print(type(train_set[0][0]))
        train_out, valid_out = lst_to_out(train_set), lst_to_out(valid_set)
        # print('\ntrain data amount: {} datas'.format(len(train_out[0])))
        # print('train_out length: {}\ntrain_out ctx shape: {}'.format(len(train_out), train_out[4].shape))
        # print('\nvalid data amount: {} datas'.format(len(valid_out[0])))
        # print('valid_out length: {}\nvalid_out ctx shape: {}'.format(len(valid_out), valid_out[4].shape))
    return train_out, valid_out


if __name__ == '__main__':
    pickle_out(amount=5, save_dir='/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/pickle')
    # amount=1, spent time: 0.45s, MEM usage rate: 1.1%
    # amount=5, spent time: 2.13s, MEM usage rate: 4.5%
    # amount=10, spent time: 9.93s, MEM usage rate: 12.5%

import os
from NeuralGraph.dataset import AllData
from NeuralGraph.processing import pickle_to_input, lst_to_out
from NeuralGraph.util import Timer


def str_key(a):
    ka = a.strip().split('_')[2]
    return ka


def main():
    save_dir = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/pickle'
    start, amount = 0, 4
    input_lst = []

    dir_lst = os.listdir(save_dir)
    dir_lst.sort(key=str_key)
    for file in dir_lst[start:start+amount]:
        file = '{}/{}'.format(save_dir, file)
        print(file)
        input_lst += pickle_to_input(file)
    out = lst_to_out(input_lst)
    print(out[0].shape)


if __name__ == '__main__':
    main()

from collections import Counter
from NeuralGraph.processing import data_parser, pd_to_pickle
from NeuralGraph.pickle_out import pickle_out


def main():
    lst = []
    with open('dataset/pre_data_all.txt','r') as f:
        lst = f.readlines()
    # tmp_lst = lst[90001:103001]
    tmp_lst = lst[:1001]
    tmp_lst = [line.strip().split('\t')[-1] for line in tmp_lst]
    print(Counter(tmp_lst))


def mm():
    data_path = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset'
    save_dir = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/pickle'
    PD_FILE = 'pre_data_all.txt' # pre_data_all.txt
    pd_all_lst = data_parser(data_path, pd_filename=PD_FILE)
    print(len(pd_all_lst))
    print(pd_all_lst[0])
    pd_all_lst = [i[-1] for i in pd_all_lst]
    print(Counter(pd_all_lst[:1001]))


# def po():
#     train_set, valid_set = pickle_out(start=1, amount=1, random_state=None)

if __name__=='__main__':
    main()
    mm()
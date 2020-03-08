from NeuralGraph.processing import data_parser, pd_to_pickle
from NeuralGraph.util import Timer


def lst_div(lst, batch_size=1000):
    lst = [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]
    return lst


def main():
    data_path = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset'
    # save_dir = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/test'
    save_dir = '/home/ubuntu/wangzhongxu/gcnn2/NGFP/dataset/pickle'
    PD_FILE = 'pre_data_all.txt' # pre_data_all.txt
    BATCH_SIZE = 1000 # 1000
    MAX_DEGREE = 6
    MAX_ATOMS = 80
    now, start = 0, 0

    pd_all_lst = data_parser(data_path, pd_filename=PD_FILE)
    all_size = len(pd_all_lst)
    pd_all_lst = lst_div(pd_all_lst, batch_size=BATCH_SIZE)
    batch_size = len(pd_all_lst)
    print('\nprocessing start: data length is {}\nfile save path: {}\n'.format(all_size, save_dir))
    for pd_lst in pd_all_lst:
        if now<start: now+=1;continue
        begin_idx, end_idx = now*BATCH_SIZE+1, now*BATCH_SIZE+len(pd_lst)
        save_file = '{}/pd_lst_{}_{}-{}'.format(save_dir, now, begin_idx, end_idx)
        with Timer() as t:
            pd_to_pickle(save_file, pd_lst, data_path, max_degree=MAX_DEGREE, max_atoms=MAX_ATOMS)
            print('processing: {}/{} ({}-{} of {})'.format(now, batch_size, begin_idx, end_idx, all_size))
        now+=1
        break


if __name__ == '__main__':
    main()

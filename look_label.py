from collections import Counter


def main():
    lst = []
    with open('dataset/pre_data_all.txt','r') as f:
        lst = f.readlines()
    # tmp_lst = lst[90001:103001]
    tmp_lst = lst[1:1001]
    tmp_lst = [line.strip().split('\t')[-1] for line in tmp_lst]
    print(Counter(tmp_lst))


if __name__=='__main__':
    main()
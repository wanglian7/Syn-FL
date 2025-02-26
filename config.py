import argparse
import numpy as np
import pickle
import torchvision.transforms as transforms
from PIL import Image
import torch

from torch.utils.data import Dataset

SERVER_ADDR= 'localhost'   # (当在真正的分布式环境中运行时，修改为服务器的IP地址)   When running in a real distributed setting, change to the server's IP address
# SERVER_ADDR= '202.114.8.21'   # (当在真正的分布式环境中运行时，修改为服务器的IP地址)   When running in a real distributed setting, change to the server's IP address
# SERVER_ADDR= '169.254.247.194'
SERVER_PORT = 51002   # 服务器端口号

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients',help='total number of clients',type=int,default=2)   # 客户端总数，默认为4
    parser.add_argument('--clients_per_round',help='number of clients trained per round;',type=int,default=2)   # 每轮训练参与的客户端数量，默认为4
    parser.add_argument('--num_round',help='number of rounds to simulate;',type=int,default=200)    # 模拟训练的轮数，默认为200
    parser.add_argument('--E',help='number of epochs when clients train on data;',type=int,default=30)   # 每轮客户端训练的轮数，默认为30
    parser.add_argument('--n',help='data samples;',type=int,default=100)   # 每个客户端训练的数据样本数，默认为100
    parser.add_argument('--lr',help='learning rate;',type=float,default=0.1)    # 学习率，默认为0.1
    parser.add_argument('--decay',help='decay for the learning rate;',type=float,default=0.995)    # 学习率衰减系数，默认为0.995

    #is non-iid
    parser.add_argument('--iid',help='is iid',type=bool,default=False)   # 是否采用非独立同分布的数据集，默认为False（即采用独立同分布数据集）
    parser.add_argument('--non_iid_degree',help='degree of non-iid',type=int,default=1)   # 非独立同分布数据集的程度，默认为1
    parser.add_argument('--num_straggle',help='number of straggler',type=int,default=0)   # 故障节点数量，默认为0
    parser.add_argument('--sleep_secs',help='sleep secs for straggler',type=int,default=10)    # 故障节点休眠时间，默认为10秒
    parser.add_argument('--seed',help='seed for randomness;',type=int,default=0)    # 随机数种子，默认为0
    parsed = parser.parse_args()     # 将解析结果赋值给变量parsed，它是一个命名空间对象，包含了所有的命令行参数及其对应的值
    options = parsed.__dict__     #  将 parsed 对象转换为字典
    return options

def read_data(data_dir):    #  从指定目录中加载 pickle 序列化的数据，然后将其转换为 MiniDataset 类型的数据对象，并将其返回
    """Parses data in given train and test data directories

    Assumes:  假设条件
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:   返回值
        clients: list of client ids    客户端ID的列表
        groups: list of group ids; empty list if none found    分组ID的列表
        train_data: dictionary of train data (ndarray)    训练数据的字典
        test_data: dictionary of test data (ndarray)    测试数据的字典
    """

    #clients = []
    #groups = []

    data = {}
    print('>>> Read data from:',data_dir)

    #open training dataset pkl files    从文件流中反序列化出一个字典 cdata，该字典包含了训练数据和标签
    with open(data_dir, 'rb') as inf:
        cdata = pickle.load(inf)
    data.update(cdata)
    data= MiniDataset(data['x'], data['y'])

    return data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target
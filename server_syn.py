import os
import copy
import time
import pickle
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.io
#from opt import find_syn_no_straggle_soln
from config import SERVER_ADDR,SERVER_PORT,read_options,read_data
import importlib
import socket
from utils import recv_msg, send_msg

# 测试推断
def test_inference(model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    #device = 'cuda' if args['gpu'] else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        #images = Variable(images.view(-1, 28 * 28))
        #labels = Variable(labels)
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/total
    return accuracy, loss
#change  平均权重
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# Model
class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        logit = self.sm(x)
        return logit

# change    选择参与训练的客户端
def select_clients():
    num_clients = min(options['clients_per_round'], n_nodes)
    return np.random.choice(range(0,len(client_sock_all)), num_clients, replace=False).tolist()

'''
从配置文件中读取选项。
与每个客户端建立连接。
初始化全局模型并设置训练参数。
遍历全局训练轮次。
为每一轮选择一部分客户端。
将当前全局模型发送给选定的客户端。
从客户端接收本地更新，对其进行平均并更新全局模型。
每一轮后执行测试并打印结果。
将最终结果（准确度、损失、时间）保存到 MATLAB 文件。
'''
if __name__== '__main__':
    options = read_options()   # 读取配置文件中的参数

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # 创建TCP套接字对象
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # 设置套接字选项，允许地址重用
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))    # 将套接字绑定到指定地址和端口
    client_sock_all = []      # 存储所有客户端套接字的列表

    num_rounds = options['num_round']  # 获取总的训练轮数
    n_nodes = options['num_clients']  # 获取客户端数量
    print("Total Clients: ", n_nodes, "run ", num_rounds, "num_rounds")  # 打印客户端数量和训练轮数
    aggregation_count = 1  # 初始化聚合计数器
    
    # Establish connections to each client, up to n_nodes clients, setup for clients    建立与每个客户端的连接，最多n_nodes个客户端
    while len(client_sock_all) < n_nodes:
        listening_sock.listen(n_nodes)  # 监听传入的连接请求
        print("Waiting for incoming connections...")
        (client_sock, (ip, port)) = listening_sock.accept()  # 接受客户端的连接请求
        print('Got connection from ', (ip, port))  # 打印客户端的IP地址和端口号
        print(client_sock)
        client_sock_all.append([ip, port, client_sock])  # 将客户端的IP地址、端口号和套接字添加到列表中

    for i in range(0, n_nodes):
        msg = ['MSG_INIT_SERVER_TO_CLIENT', options, i]  # 构造初始化消息
        send_msg(client_sock_all[i][2], msg)  # 发送消息给客户端

    print('All clients connected')
   
    # test_dataset = []
    test_data = read_data('./MNIST_test.pkl')    # 从文件中读取测试数据
    #test_loader = DataLoader(dataset=test_data,
    #                         batch_size=128,
    #                         shuffle=True)
    #test_data = dsets.MNIST(root = './mnist',
    #                        train = False,
    #                        transform = transforms.ToTensor(),
    #                        download = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size = 128,shuffle = True)   # 创建测试数据加载器

    #  开启全局训练
    #mnist
    #global_model = Logistic(28, 5)
    global_model = Logistic(784,10)   # 创建全局模型
    #global_model.to(device)
    global_model.train()    # 设置模型为训练模式
    global_weights = global_model.state_dict()  # 获取全局模型的权重参数
    train_accuracy, train_loss = [], []  # 存储训练准确率和损失
    cv_loss, cv_acc, cv_time = [], [], []  # 存储交叉验证损失、准确率和时间
    print_every = 2  # 每隔2轮打印一次信息

    #initial parameters    初始化参数
    E_train = options['E']  # 获取每轮训练的步数
    n_train = options['n']  # 获取每个客户端的训练样本数
    start = time.time()  # 记录开始时间
    done = False  # 是否完成标志
    while not done:
        print(f'\n | Global Training Round : {aggregation_count} |\n')

        global_weights = global_model.state_dict()   # 获取当前全局模型的权重
        #first test the full partition
        selected_clients = select_clients()  # 选择参与训练的客户端
        is_last_round = False  # 是否是最后一轮的标志
        print('---------------------------------------------------------------------------')
        aggregation_count += 1  # 聚合计数器加1

        # test   向每个客户端发送全局权重
        for i in range(n_nodes):
            msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights,E_train,n_train]
            send_msg(client_sock_all[i][2], msg)    # 发送消息给客户端

        print('Waiting for local iteration at client')

        local_weights = []  # 存储每个客户端的本地权重
        for i in range(n_nodes):
            msg = recv_msg(client_sock_all[i][2], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')  # 接收消息
            w = msg[1]  # 提取本地权重
            local_weights.append(copy.deepcopy(w))  # 将本地权重添加到列表中

        global_weights = average_weights(local_weights)  # 计算平均权重
        global_model.load_state_dict(global_weights)  # 更新全局模型的权重
        test_acc, test_loss = test_inference(global_model, test_loader)  # 进行测试推断
        print(test_acc, test_loss)  # 打印测试准确率和损失
        cv_acc.append(test_acc)  # 存储测试准确率
        cv_loss.append(test_loss)  # 存储测试损失
        latency = time.time() - start  # 计算耗时
        cv_time.append(latency)  # 存储耗时
        if test_acc >= 0.885:
            done = True  # 如果达到指定准确率，则完成训练
    num_straggle=options['num_straggle']    # 获取延迟客户端数量
    if options['iid']:
        saveTitle = 'SYN_iid_slow' + str(num_straggle) + '_' + str(E_train) + '_' + str(n_train)  # 构造保存文件名
    else:
        saveTitle = 'SYN_niid_slow' + str(num_straggle) + '_' + str(E_train) + '_' + str(n_train)  # 构造保存文件名
    scipy.io.savemat(saveTitle + '_acc' + '.mat', mdict={saveTitle + '_acc': cv_acc})  # 保存交叉验证准确率数据
    scipy.io.savemat(saveTitle + '_loss' + '.mat', mdict={saveTitle + '_loss': cv_loss})  # 保存交叉验证损失数据
    scipy.io.savemat(saveTitle + '_time' + '.mat', mdict={saveTitle + '_time': cv_time})  # 保存交叉验证耗时数据
    # Save tracked information
    print("Total number of training rounds："+str(aggregation_count))



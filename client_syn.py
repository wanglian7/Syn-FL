'''
客户端的主要功能：
（1）接收服务器下发的指令和全局模型
（2）利用本地数据进行局部模型训练
'''
import torch.nn as nn
import torch
import pickle #for pkl file reading
import os
import sys
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset,SubsetRandomSampler
#from PIL import Image
import torchvision.transforms as transforms
import time
from config import SERVER_ADDR, SERVER_PORT,read_data,read_options
from utils import recv_msg, send_msg
import socket
import struct

# Model for MQTT_IOT_IDS dataset   定义MQTT_IOT_IDS数据集的模型
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

###################################################################################

# socket    创建 socket 连接
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
print('---------------------------------------------------------------------------')
try:
	# （1）客户端从服务器接收初始化消息，其中包括学习率、权重衰减、客户端数量、客户端ID等选项。
	msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
	options = msg[1]
	n_nodes=options['num_clients']      # 客户端数量
	cid = msg[2]     # 当前客户端ID
	#first step: set the optimizer & criterion   设置优化器和损失函数
	lr_rate=options['lr']    # 学习率
	gamma=options['decay']   # 权重衰减
	num_straggle=options['num_straggle']    # 延迟执行的客户端数量
	is_iid=options['iid']    # 数据集是否为独立同分布（IID）
	sleep_secs=options['sleep_secs']     # straggler 的休眠时间
	print("Receive the Learning rate:",lr_rate," Weight_decay:",gamma)

    # （2） 根据接收到的选项，客户端设置优化器、学习率调度器和损失函数。
	model=Logistic(784,10)      # 创建模型
	optimizer=torch.optim.SGD(model.parameters(),lr=lr_rate)    # 创建优化器
	scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma,last_epoch=-1)    # 创建学习率衰减策略
	criterion = torch.nn.CrossEntropyLoss()    # 创建损失函数
	
	# （3） Import the data set    导入数据集
	print("read data")
	if is_iid:
		train_data = dsets.MNIST(root='./mnist',train = True,transform = transforms.ToTensor(),download = True)
		print("succ read iid!!!!")
	else:
		non_iid_degree=options['non_iid_degree']    # 非独立同分布（NIID）的程度
		file_path='./niid'+str(non_iid_degree)+'/mnist'+str(cid)+'.pkl'    # 数据集文件路径
		train_data=read_data(file_path)    # 读取数据集
		print("succ read non-iid from",file_path)

	'''
     （4）主训练循环：
	    客户端进入一个循环，不断接收来自服务器的全局模型权重、轮次等信息
	    构建用于训练数据的数据加载器
	    将全局模型权重加载到本地模型中  (全局模型是联邦学习系统中的中央模型，其权重由所有参与方的本地模型共同学习得到)   (全局模型权重指的是在联邦学习中的中央模型所包含的参数值)
	    客户端执行本地训练（前向传播、反向传播和优化器步骤）指定次数的轮次
	    训练后，客户端将本地模型权重和其他信息发送回服务器
	'''
	while True:
		print('---------------------------------------------------------------------------')
	    # 接收来自服务器的权重和轮次消息
		msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
		is_last_round = msg[1]    # 是否为最后一轮
		global_model_weights = msg[2]     # 全局模型权重
		num_epoch = msg[3]    # 迭代轮次
		n = msg[4]    # 数据样本数量
		print('Epoches: ',num_epoch,' Number of data samples: ',n)

		# make the data loader     创建数据加载器
		train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=n,shuffle=True)
		print('Make dataloader successfully')

		model.load_state_dict(global_model_weights)    # 加载全局模型权重

		#num_epoch = options['num_epoch']  # Local epoches
		#batch_size = 400  # Data sample for training per comm. round

		model.train()    # 设置模型为训练模式

		#print('Round:', round)
		for i in range(num_epoch):
			x, y = next(iter(train_loader))
			if is_iid:
				x = x.view(x.size(0), -1) # x = Variable(x.view(-1,28*28))
				y = Variable(y)
			optimizer.zero_grad()  # 梯度清零
			pred = model(x)  # 前向传播
			loss = criterion(pred, y)  # 计算损失
			loss.backward()  # 反向传播
			optimizer.step()  # 更新模型参数

		scheduler.step()    # 更新学习率
		print(optimizer.state_dict()['param_groups'][0]['lr'])
		if cid<num_straggle:
			print("I am straggle sleep for",sleep_secs," s")
			time.sleep(sleep_secs)
        # sleep 10s for straggler
		# acc, loss = local_test(model=model, test_dataloader=test_loader)

		msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', model.state_dict()]    # 发送模型权重和时间信息给服务器
		send_msg(sock, msg)
		#print("loss:", loss, "    acc:", acc)
		if is_last_round:
			break

except (struct.error, socket.error):
    print('Server has stopped')
    pass


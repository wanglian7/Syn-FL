# Syn-FL

使用数据集为MNIST数据集（也可使用其他数据集）。

在本地模拟，令config.py文件中的 SERVER_ADDR= 'localhost'

分布式多设备模拟，令config.py文件中的 SERVER_ADDR = '192.168.1.100'   即将ip改为协调器ip地址即可。

先运行server.py文件，后对client_asyn.py文件进行任务多开即可分布式运行此同步联邦学习代码。

从网盘链接 https://pan.baidu.com/s/1yjCV9hIBEjyykSP7pfsF7A?pwd=3mg3 （提取码: 3mg3）下载数据集文件，放在niid1目录下，代码即可运行。

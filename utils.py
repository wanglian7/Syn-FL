import pickle, struct, socket

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())

# def recv_msg(sock, expect_msg_type=None):
#     msg_len = struct.unpack(">I", sock.recv(4))[0]
#     msg = sock.recv(msg_len, socket.MSG_WAITALL)
#     msg = pickle.loads(msg)
#     print(msg[0], 'received from', sock.getpeername())

def recv_msg(sock, expect_msg_type=None):
    # 首先接收 4 字节的消息长度
    msg_len_data = b''
    while len(msg_len_data) < 4:
        msg_len_data += sock.recv(4 - len(msg_len_data))

    msg_len = struct.unpack(">I", msg_len_data)[0]

    # 接收消息数据
    msg_data = b''
    while len(msg_data) < msg_len:
        chunk = sock.recv(msg_len - len(msg_data))
        if not chunk:
            # 处理连接关闭或其他错误的情况
            break
        msg_data += chunk

    msg = pickle.loads(msg_data)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


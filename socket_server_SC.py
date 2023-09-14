import socket
import pickle
import numpy as np
import struct

"""
socket 통신으로 server가 client에게 numpy array를 수신

"""


def socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    localhost = "116.36.30.109"
    server_socket.bind((localhost, 3579))

    server_socket.listen(1)
    print("서버가 시작되었습니다. 연결을 기다리는 중...")

    client_socket, addr = server_socket.accept()
    print(f"클라이언트가 연결되었습니다. {addr}")

    # 클라이언트로부터 데이터 사이즈 수신
    data_size_buffer = client_socket.recv(4)
    data_size = struct.unpack("!I", data_size_buffer)[0]

    data_buffer = b""
    while len(data_buffer) < data_size:
        recv_data = client_socket.recv(1024)
        if not recv_data:
            break
        data_buffer += recv_data
        print(len(data_buffer))

    # 수신된 데이터 역직렬화
    recieved_array = pickle.loads(data_buffer)
    # 소켓 닫기
    client_socket.close()
    server_socket.close()

    return recieved_array

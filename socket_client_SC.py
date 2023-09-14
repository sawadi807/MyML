import socket
import numpy as np
import pickle
import struct

"""
socket 통신으로 client가 server에게 numpy array를 전송
"""


# Create a TCP/IP socket
def socket_client(numpy_array):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    localhost = "116.36.30.109"
    # Connect the socket to the server's address and port
    server_address = (localhost, 3579)
    client_socket.connect(server_address)

    serialized_data = pickle.dumps(numpy_array)

    # 데이터 크기 전송
    data_size = len(serialized_data)
    client_socket.sendall(struct.pack("!I", data_size))

    print(data_size)

    client_socket.sendall(serialized_data)

    print("데이터가 전송되었습니다.\n")

    message = client_socket.recv(1024)
    message = message.decode()
    if message == "데이터 전송 완료":
        print("데이터 수신 확인")
    client_socket.close()

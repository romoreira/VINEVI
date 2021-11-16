import socket 

def main():
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3)) 
    while True: 
        raw_data, _ = s.recvfrom(65535)
        print(raw_data[14:])

main()
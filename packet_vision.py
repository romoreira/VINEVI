import numpy as np
from PIL import Image
import pandas as pd
import sys

def create_image(packet, packet_number):
    matrix = []
    j = 8
    i = 0
    i_bkp = 0
    print("Packet to be created: "+str(packet))
    while True:
        print("I: " + str(i))
        print("I + J: "+str(i + j))
        matrix.append(packet[i: (i + j)])
        i_bkp = i
        i = (i + 8) + 1
        if((i + j) > len(packet)):
            j = ((i_bkp + j)) + len(packet) - (i_bkp + j)
            l = [packet[i: (i + j)]]
            append = 8 - len(l[0])
            for m in range(8 - len(l[0])):
                l[0].insert(append,'FF')
                append = append + 1
                print(l[0])
            matrix.append(l[0])
            print("I: " + str(i))
            print("J: " + str(j))
            break
        print("J: "+str(j))
    print("Matrix original: "+str(matrix))

    matrix = pd.DataFrame(matrix)
    matrix = matrix.to_numpy()

    #print("Matrix teste: "+str(matrix))

    matrix = np.matrix(matrix)

    shape = matrix.shape[0]

    for i in range(matrix.shape[0]):
        for j in range(8):
            matrix[i, j] = str(int(matrix[i, j], 16))
    print(matrix)
    matrix = matrix.tolist()
    print(matrix)

    for i in range(shape):
        for j in range(8):
            matrix[i][j] = [matrix[i][j],matrix[i][j],matrix[i][j]]
    print(matrix)

    data = np.array(matrix)
    img = Image.fromarray(data.astype('uint8'), 'RGB')
    size=shape*8
    print("\nSaving packet_" + str(packet_number))
    img.save("packet_"+str(packet_number)+".png")
    return

def process_pipe(data_from_pipe, packet_number):
    print("Printing from Python")
    print("Data_from_pipe: "+str(data_from_pipe))
    data_from_pipe = data_from_pipe.rstrip("\n")
    data_from_pipe = list(data_from_pipe.split(" "))
    print("Data_from_pipe 'enter' not removed: "+str(data_from_pipe))
    data_from_pipe.pop()
    print("Data_from_pipe 'enter' removed: " + str(data_from_pipe))
    create_image(data_from_pipe, packet_number)


def main():
    packet_number = 0
    for line in iter(sys.stdin.readline, " "):
        packet_number = packet_number + 1
        process_pipe(line, packet_number)

    # with open('raw_packet_temp') as f:
    #     lines = f.readlines()
    #     print("Lines no comeco: "+str(lines))
    #     lines = lines[0]
    #     lines = lines.split(' ')
    #     print(lines)
    #     create_image(lines)

if __name__ == '__main__':
    main()
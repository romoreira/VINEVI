# Link dataset: https://cba.upc.edu/downloads/category/24-traces#

# FB dataset: http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/

import numpy as np
from PIL import Image
import pandas as pd
import re

def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def process_text():
    print("Inicio")

    lines = ""
    with open('xaa') as f:
        lines = f.readlines()

    i = 0#Comeca sem nada
    string = ''
    n = 0
    lst = []
    count = 0
    x = 0;
    y = 0
    for each in lines:
        print("Linha arquivo: "+str(each))
        if i == 1:
            if "};" in each:
                print("Lista final: "+str(lst))


                l = lst[-1]
                print(l)
                lst = lst[:-1]

                acrescimo = 8 - len(l)
                print("Acrescimo: "+str(acrescimo))
                for i in range(acrescimo):
                    l.insert(len(l), "0xFF")

                lst.append(l)

                print("FINAL LIST: "+str(lst))

                create_image(lst,count,x)
                x = 0
                lst = []
                l = []
                i = 0
                count = count + 1
                print("Continua...: "+str(count))

            else:
                #print("Linha nao-final: "+str(each))
                each=comment_remover(each)
                #print("After comments removed: "+str(each))
                each=each.replace(",", "")
                #print("Each removed blank spaces: "+str(each))
                #print("Split in nao-final "+str(each.split("\n")))

                l = each.split(" ")
                #print("Lista final: "+str(l))
                #print("len(l): "+str(len(l)))

                if len(l) > 8:
                    print(l)
                    l.pop(8)
                    print(l)
                    l.pop(8)
                print("lista final apos pop: "+str(l))


                #print("Tamanho lista a ser removida: "+str(len(l)))
                if len(l) > 8:
                    #print("Tamanho lista a ser removida: " + str(len(l)))
                    jj = 0
                    for ii in range(len(l)):
                        #print("Each element in sublist: "+str(l[ii]))
                        if l[ii] == "" or l[ii] == "\n" or l[ii] == "*/\n":
                            print("Remover espaços e enters")
                            jj = ii
                            print("Valor de JJ: "+str(jj))
                            while len(l) > jj:
                                l = l[:-1]
                                print(l)
                                print(len(l))
                                print("JJ value: "+str(jj))
                            break


                print("Resultado ultimo elemento linha nao-final: "+str(l))
                lst.append(l)
                print(str(n+1) + " Line: "+str(string))
                n = n + 1
                x = x + 1

        if i == 0:
            if "{" in each:
                i = 1

def create_image(lst, n,x):

    print("Tamanho de X: "+str(x))

    #print(lst)
    #exit()

    #print(string)
    teste = np.asmatrix(lst)
    print(teste)
    #print(np.size(teste))
    #print(np.shape(teste))
    #print(teste[20,5])
    #print(int(teste[0,0],16))

    #if np.size(teste,1) > 169 and np.size(teste,1) <=196:
    #    print("Criar uma matriz 14 x 14")

    for i in range(x):
        for j in range(8):
            #print(str(teste[i,j]))
            teste[i,j] = int(teste[i,j],16)

    teste = teste.tolist()
    #print("Teste: "+str(teste))
    #print(teste[0][1])


    numeros = np.matrix(teste)
    numeros = numeros.astype(int)
    #print(numeros.shape)
    #print(numeros)

    dataFrame = pd.DataFrame(numeros)
    data = dataFrame.to_numpy()

    data = data.tolist()
    #print(data[0][7])

    for i in range(x):
        for j in range(8):
            data[i][j] = [data[i][j],data[i][j],data[i][j]]

    data = np.array(data)
    #print(data)


    img = Image.fromarray(data.astype('uint8'), 'RGB')
    size=n*8
    #arr = np.zeros((size,size,3))
    #arr[:,:,0] = [[255]*size]*size
    #arr[:,:,1] = [[255]*size]*size
    #arr[:,:,2] = [[0]*size]*size
    #img = Image.fromarray(arr.astype('uint8'), 'RGB')

    print("\nPronto pra salvar: " + str(n))
    img.save("facebook-video"+str(n)+".png")
    return

process_text()
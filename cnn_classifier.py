# -*- coding: utf-8 -*-
"""
Packet Vision
"""

# Importa bibliotecas necessárias
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision.datasets as datasets
import os
import random
import numpy as np

import torch.onnx as onnx


import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import seaborn as sns

import time

torch.cuda.empty_cache()

data_dir = './dataset/'

classes = ['bittorrent', 'dns', 'IoT', 'voip']

# Escolhe o modelo: resnet, alexnet, vgg, densenet, squeezenet
model_name = "squeezenet"

# Número de classes do conjunto de dados: [Abnormal, Normal]
num_classes = 4
# Tamanho do lote para treinamento
batch_size = 32
num_workers = 4

# Número de épocas para treino
num_epochs = 30

# Extrator de recursos. (False: ajuste do modelo inteiro; True: atualiza apenas os parâmetros da camada remodelada)
feature_extract = True

# Optimizer e Criterion
lr = 0.001
momentum = 0.9

criterion = nn.CrossEntropyLoss()

# Definindo as transformações para o conjunto de treino e validação.
# Definindo uma transformação para pré-processar as imagens de treinamento.

train_transforms = transforms.Compose([
    transforms.Resize(size=[224, 224]),
    # transforms.Resize(size=[299,299]),
    transforms.RandomRotation([0, 360]),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transforms = transforms.Compose([
    transforms.Resize(size=[224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trains_dir = []
valids_dir = []
train_loaders = []
valid_loaders = []
test_loaders = []

folds = os.listdir(data_dir + '5-fold')
folds.sort()

all_size_train = []
all_size_valid = []

if __name__ == "__main__":

    for i in folds:
        train_dir = os.path.join(data_dir + '5-fold/', i + '/train/')
        valid_dir = os.path.join(data_dir + '5-fold/', i + '/test/')

        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

        test_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)

        print("----------------------------------------------------------------------------------------")
        print(i)
        print('Num training images: ', len(train_data), train_loader)
        print('Num valid images: ', len(valid_data), valid_loader)
        print('Num test images: ', len(valid_data), (test_loader))

        all_size_train.append(len(train_data))
        all_size_valid.append(len(valid_data))

    print("----------------------------------------------------------------------------------------")
    print("\n\n----------------------------------------------------------------------------------------")
    print("Num train full size:", sum(all_size_train))
    print("Num valid full size:", sum(all_size_valid))
    print('Num test images: ', len(valid_data), (test_loader))
    print("Num full size (train+valid):", sum(all_size_train) + sum(all_size_valid) + len(valid_data))

    # list = [ x[0] for x in iter(test_loader).next() ]
    # print(list)
    '''
    for xb, yb in test_loader:
        print(xb.shape)
        #x = xb.view(28,28) 
        #print(x.shape)
        print(xb)
        break #just once
    '''

    batch = next(iter(test_loader))
    print('len:', len(batch))
    images, labels = batch
    print('types:', type(images), type(labels))
    print('shapes:', images.shape, labels.shape)
    print(images[0].shape)
    print(labels[0])

    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15, 15))
    # plt.imshow(np.transpose(grid, (1,2,0)))
    plt.imshow(grid.permute(1, 2, 0))
    print('labels:', labels)

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    GPUavailable = torch.cuda.is_available()
    if GPUavailable:
        print('Treinamento em GPU!')
        device = torch.device("cuda:0")
    else:
        print('Treinamento em CPU!')
        device = "cpu"

    torch.backends.cudnn.deterministic = True

    # Carregando uma rede pré-treinada, treinando e computando a acurácia de validação para cada época

    '''
    This helper function sets the .requires_grad
    attribute of the parameters in the model to False when we are feature extracting. 
    By default, when we load a pretrained model all of the parameters have .requires_grad=True, 
    which is fine if we are training from scratch or finetuning. 
    However, if we are feature extracting and only want to compute gradients 
    for the newly initialized layer then we want all of the other parameters 
    to not require gradients. This will make more sense later.
    '''


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = True


    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Inicializando cada variável específica para cada modelo
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            # model_ft = models.resnet18(pretrained=use_pretrained)
            model_ft = models.resnet34(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224


        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet169(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "mobilenet":
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "shufflenet":
            model_ft = models.shufflenet_v2_x1_0(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224


        elif model_name == "inception":
            """ Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained, aux_logits=False)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


    # Inicializa o modelo para esta execução
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Printa o modelo instanciado
    # print(model)

    torch.cuda.empty_cache()

    # Envia o modelo para a GPU
    model = model.to(device)

    # Reúne os parâmetros a serem otimizados/atualizados nesta execução.
    # Caso ajustado, estaremos atualizando todos os parâmetros.
    # Caso, usando método de extração de recursos, atualiza apenas os parâmetros que acabamos de inicializar,
    # ou seja, os parâmetros com require_grad são "True".

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Optimizer
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
    # optimizer = optim.Adam(params_to_update, lr = FOUND_LR)

    # Listas --------------------------
    train_correct_list = []
    train_predict_list = []

    valid_correct_list = []
    valid_predict_list = []

    test_correct_list = []
    test_predict_list = []


    # ----------------------------------

    def conf_matrix(fx, y, nome):

        if (nome == 'treino'):
            preds = fx.max(1, keepdim=True)[1]
            correct = y

            c = correct.tolist()
            p = preds.flatten().tolist()

            train_correct_list.append(c)
            train_predict_list.append(p)

            return train_correct_list, train_predict_list

        if (nome == 'validacao'):
            preds = fx.max(1, keepdim=True)[1]
            correct = y

            c = correct.tolist()
            p = preds.flatten().tolist()

            valid_correct_list.append(c)
            valid_predict_list.append(p)

            return valid_correct_list, valid_predict_list

        if (nome == 'teste'):
            preds = fx.max(1, keepdim=True)[1]
            correct = y

            c = correct.tolist()
            p = preds.flatten().tolist()

            test_correct_list.append(c)
            test_predict_list.append(p)

            return test_correct_list, test_predict_list


    def calculate_accuracy(fx, y):
        preds = fx.max(1, keepdim=True)[1]
        correct = preds.eq(y.view_as(preds)).sum()
        acc = correct.float() / preds.shape[0]

        return acc


    def train(model, device, iterator, optimizer, criterion, nome):

        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            # Matriz de Confusão. Recebe os dados e gera listas
            c, p = conf_matrix(fx, y, nome)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator), c, p


    def evaluate(model, device, iterator, criterion, nome):

        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for (x, y) in iterator:
                x = x.to(device)
                y = y.to(device)

                fx = model(x)

                loss = criterion(fx, y)

                acc = calculate_accuracy(fx, y)

                # Matriz de Confusão. Recebe os dados e gera listas
                c, p = conf_matrix(fx, y, nome)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator), c, p


    # Cria a pasta "./Resultados"
    # Cria a pasta "./Resultados/"model_name"

    name = './Resultados'
    if os.path.isdir(name) == False:
        os.mkdir(name)

    resultados_dir = './Resultados/' + model_name
    if os.path.isdir(resultados_dir) == False:
        os.mkdir(resultados_dir)

    # Lista para calcular a media da validação e teste
    media_val = []
    media_test = []

    def model_saving(model, MODEL_SAVE_PATH):
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("CNN Model Saved on Disk")

    def train_function(i, x):
        train_losses = []
        val_losses = []

        train_accuracy = []
        val_accuracy = []

        EPOCHS = num_epochs
        SAVE_DIR = 'models_training'
        MODEL_SAVE_PATH = os.path.join(SAVE_DIR, model_name + '-endovis.pth')

        best_valid_loss = float('inf')

        if not os.path.isdir(f'{SAVE_DIR}'):
            os.makedirs(f'{SAVE_DIR}')

        f = open(resultados_dir + '/' + x + '.csv', 'w',
                 newline='')  # Cria o arquivo .csv baseado no round em execução (ex: round_1.csv)
        writer = csv.writer(f)

        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val. Loss', 'Val. Acc'])

        for epoch in range(EPOCHS):

            train_loss, train_acc, train_correct, train_pred = train(model, device, train_loaders[i], optimizer,
                                                                     criterion, 'treino')


            valid_loss, valid_acc, valid_correct, valid_pred = evaluate(model, device, valid_loaders[i], criterion,
                                                                        'validacao')

            # Pegar somente a ultima Época para relatório da matriz de confusão
            if (epoch != (EPOCHS - 1)):
                valid_correct.clear()
                valid_pred.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_saving(model, MODEL_SAVE_PATH)

            print(
                f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:05.2f}% |')
            writer.writerow([(epoch + 1), (train_loss), (train_acc * 100), (valid_loss),
                             (valid_acc * 100)])  # Escreve os dados no arquivo .csv

            train_losses.append(train_loss)
            val_losses.append(valid_loss)

            train_accuracy.append(train_acc)
            val_accuracy.append(valid_acc)



        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        # test_loss, test_acc = evaluate(model, device, test_loader, criterion, 'teste')
        test_loss, test_acc, test_correct, test_preds = evaluate(model, device, test_loaders[i], criterion, 'teste')

        # Matriz de Confusão para teste
        print("===========================================================================================")
        matriz_confusao(test_correct, test_preds, 'teste', x)
        print("===========================================================================================")

        print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:05.2f}% |')

        # Escreve os dados no arquivo .csv
        writer.writerow(['', '', '', '', ''])
        writer.writerow(['Test Loss', 'Test Acc', '', '', ''])
        writer.writerow([(test_loss), (test_acc * 100), '', '', ''])

        # Guarda os resultados finais de validação e teste dos folds para posteriormente calcular a média
        media_val.append(valid_acc * 100)
        media_test.append(test_acc * 100)

        f.close()  # Fecha arquivo .csv

        #   %matplotlib inline
        #   %config InlineBackend.figure_format = 'retina'

        # Cria os graficos de decaimento treino e validação (imprime na tela e salva na pasta "./Resultados")
        plt.title(x)
        plt.plot(train_losses, label='Loss')
        # plt.plot(val_losses, label='Validation loss')
        plt.plot(train_accuracy, label='Accuracy')
        plt.legend(frameon=False)
        plt.grid()
        plt.savefig(resultados_dir + '/' + 'graf_' + x + '.png')
        plt.close()
        # plt.show()
        '''
        plt.title(x)
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig(resultados_dir+'/'+'graf_'+x+'.png')
        plt.close()
        #plt.show()
  
        # Cria os graficos de acurácia do treino e validação (imprime na tela e salva na pasta "./Resultados")
        plt.title(x)
        plt.plot(train_accuracy, label='Training accuracy')
        plt.plot(val_accuracy, label='Validation accuracy')
        plt.legend(frameon=False)
        plt.savefig(resultados_dir+'/'+'grafAcc_'+x+'.png')
        plt.close()
        #plt.show()
        '''


    def clear_list():
        train_correct_list.clear()
        train_predict_list.clear()

        valid_correct_list.clear()
        valid_predict_list.clear()

        test_correct_list.clear()
        test_predict_list.clear()


    # Matriz de Confusão
    def matriz_confusao(correct, pred, nome, x):
        correct_list = []
        predict_list = []

        for i in correct:
            correct_list.extend(i)

        for j in pred:
            predict_list.extend(j)

        print("Listas: ")
        print("Correct: ", correct_list)
        print("Predict: ", predict_list)
        print("\n")

        print("Matriz de Confusão (" + nome + ") do " + x + ": ")
        cm = confusion_matrix(correct_list, predict_list)
        print(cm)

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")  # annot=True to annotate cells
        ax.set_title('Matriz de Confusão (' + nome + ') (' + x + ')')
        plt.savefig(resultados_dir + '/' + 'matrizconfusao_' + nome[:3] + '_' + x + '.png')
        plt.close()
        plt.show()

        print("\nRelatório de classificação (" + nome + "): ")
        report = metrics.classification_report(correct_list, predict_list, target_names=classes, digits=4)
        print(metrics.classification_report(correct_list, predict_list, target_names=classes))

        file_report = open('./Resultados/' + model_name + '/report_matrix.txt', 'a+')
        file_report.write("%s \n" % nome)
        file_report.write("%s \n" % x)
        file_report.write(report)
        file_report.write("\n")
        file_report.write("Matriz de Confusão \n")
        file_report.write(str(cm))
        file_report.write("\n \n")
        file_report.close()

        '''
         # calculate the fpr and tpr for all thresholds of the classification
        fpr, tpr, threshold = metrics.roc_curve(correct_list, predict_list)
        roc_auc = metrics.auc(fpr, tpr)
  
        # method I: plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(resultados_dir+'/'+'ROC_'+nome[:3]+'_'+x+'.png')
        plt.close() 
        #plt.show()
        '''


    def calc_time(time_total):
        segundos = time_total

        segundos_rest = segundos % 86400
        horas = segundos_rest // 3600
        segundos_rest = segundos_rest % 3600
        minutos = segundos_rest // 60
        segundos_rest = segundos_rest % 60

        plot_time = "%d hours" % horas + ", %d minutes" % minutos + ", %d seconds" % segundos_rest
        return plot_time


    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)
            torch.nn.init.xavier_uniform(m.bias.data)


    # Executando a função de treino para cada fold e calcula média (validação e treino)

    for i in range(len(folds)):
        print("===========================================================================================")
        print("Inicialização do Particionamento, ", folds[i])
        print("===========================================================================================")

        clear_list()  # A cada execução do laço de repetição as listas que coletam as preds e corrects são limpadas
        start = time.time()
        train_function(i, str(folds[i]))
        end = time.time()
        time_total = (end - start)

        plot_time = calc_time(time_total)
        print(plot_time)

        tempo_txt = open('./Resultados/' + model_name + '/tempo_treinamento.txt', 'a+')
        tempo_txt.write(folds[i] + ": ")
        tempo_txt.write(plot_time + "\n")
        tempo_txt.close()
        print("===========================================================================================")
        print("\n")

    # Média do k-fold
    print(media_val)
    kfold = 0
    for i in range(len(media_val)):
        kfold = kfold + media_val[i]

    media_kfold_val = kfold / (len(media_val))

    print('\nVal Accuracy (Median): %.2f%%' % (media_kfold_val))

    print(media_test)
    kfold = 0
    for i in range(len(media_test)):
        kfold = kfold + media_test[i]

    media_kfold_test = kfold / (len(media_test))

    print('\nTest Accuracy (Median): %.2f%%' % (media_kfold_test))

    m = open(resultados_dir + '/' + 'medias.csv', 'w', newline='')
    writer = csv.writer(m)

    writer.writerow(['Val Accuracy', 'Test Accuracy'])
    writer.writerow([(media_kfold_val), (media_kfold_test)])
    m.close()

    # Limpa listas para possíveis novas execuções em células
    media_val.clear()
    media_test.clear()
import torch
from torchvision.transforms import transforms
from PIL import Image
from torchvision import datasets, models, transforms
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import sys
from csv import writer
import threading
import time


SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("CPU-based classification")

time_list = []

know_classes = ['bittorrent', 'browsing','dns', 'iot', 'rdp', 'ssh','voip']

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Inicializando cada variável específica para cada modelo
    model_ft = None
    input_size = 0

    if model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "mobilenet":
      model_ft = models.mobilenet_v2(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.classifier[1].in_features
      model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
      input_size = 224

    elif model_name == "alexnet":
      model_ft = models.alexnet(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.classifier[6].in_features
      model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
      input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True


def cnn_start():
    model = 0
    input_size = 0
    model_name = "alexnet"
    #print("Initializing CNN Model...")
    model, input_size = initialize_model(model_name, num_classes=7, feature_extract=True, use_pretrained=True)

    # print("Model before load: \n"+(str(model)))

    checkpoint = torch.load(Path('/home/ubuntu/VINEVI/models_training/alexnet.pth'), map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    #get_cnn_complexity(model)

    return model

def get_cnn_complexity(model):
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def write_csv(register):
    with open(str(sys.argv[2])+'_exp_time_spent_on_prediction.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(register)
        f.close()

def cnn_predict(image_name, class_to_test):
    # print("Imagem type: "+str(type(image)))
    model = cnn_start()

    test_transforms = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    path = Path('/home/ubuntu/VINEVI/images_test/'+str(class_to_test)+'/'+str(image_name))
    #print("Caminho do load image: "+str(path))

    image = Image.open(Path('/home/ubuntu/VINEVI/images_test/'+str(class_to_test)+'/'+str(image_name)))
    # print("Image Type Load: "+str(type(image)))

    input = test_transforms(image)
    #print('types:', type(input))
    #print("Transformd Image: " + str(input.shape))
    input = torch.unsqueeze(input, 0)
    # print("Types last: "+str(type(input)))


    before_predict_time = time.time_ns()
    output = model(input)
    after_predict_time = time.time_ns()
    time_list.append(before_predict_time)
    time_list.append(after_predict_time)
    time_list.append(after_predict_time - before_predict_time)
    

    prediction = output.max(1, keepdim=True)[1]
    #print("Prediction: "+str(prediction))
    #print(know_classes[int(prediction.item())])

    # print(model)

    time_list.append(know_classes[int(prediction.item())])
    time_list.append(image_name)
    write_csv(time_list)

    return know_classes[int(prediction.item())]

# def service():
#    print("Waiting for new images")
#    while(True):
#        #print("Waiting for")
#        a = 1

# def main():
#    model = cnn_start()
#    thread1 = threading.Thread(target=service)
#    thread1.start()
#    thread2 = threading.Thread(target=cnn_predict,args=(model,))
#    thread2.start()

if __name__ == '__main__':
    none = ""
    #print("Predicted: "+str(cnn_predict(sys.argv[1], sys.argv[2])))
    cnn_predict(sys.argv[1], sys.argv[2])

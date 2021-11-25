import torch
from torchvision.transforms import transforms
from PIL import Image
from torchvision import datasets, models, transforms
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn



know_classes = ['bittorrent', 'dns', 'IoT', 'voip']

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
    model_name = "squeezenet"
    print("Initializing CNN Model...")
    model, input_size = initialize_model(model_name, num_classes=4, feature_extract=True, use_pretrained=True)

    # print("Model before load: \n"+(str(model)))

    checkpoint = torch.load(Path('/home/ubuntu/vinevi/models_training/squeezenet-endovis.pth'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def cnn_predict(image):
    # print("Imagem type: "+str(type(image)))
    model = cnn_start()
    train_transforms = transforms.Compose([
        transforms.Resize(size=[224, 224]),
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

    # image = Image.open(Path('/home/ubuntu/vinevi/video1235.png'))
    # print("Image Type Load: "+str(type(image)))

    input = train_transforms(image)
    print('types:', type(input))
    print("Transformd Image: " + str(input.shape))
    input = torch.unsqueeze(input, 0)
    # print("Types last: "+str(type(input)))

    output = model(input)
    # print("Output: \n"+str(type(output)))

    prediction = output.max(1, keepdim=True)[1]
    # print(prediction)

    # print(model)

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

# if __name__ == '__main__':
#    main()
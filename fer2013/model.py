from torchvision import models
import torch.nn as nn
import torch.functional as F

# feature_extract is a boolean that defines if we are finetuning or feature extracting. 
# If feature_extract = False, the model is finetuned and all model parameters are updated. 
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


class CustomNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.Conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1)
        self.Conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        
        self.Conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        self.Conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.Conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1)
        self.Conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        
        self.Conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1)
        self.Conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        
        self.fc1 = nn.Linear(512*3*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,7)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.5)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.relu(self.Conv1_1(x))
        x = F.relu(self.Conv1_2(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = F.relu(self.Conv2_1(x))
        x = self.bn2(x)
        x = F.relu(self.Conv2_2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = F.relu(self.Conv3_1(x))
        x = self.bn3(x)
        x = F.relu(self.Conv3_2(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = F.relu(self.Conv4_1(x))
        x = self.bn4(x)
        x = F.relu(self.Conv4_2(x))
        x = self.bn4(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = x.view(-1,512*3*3)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        
        x = self.fc4(x)
        
        output = self.softmax(x)
        
        return output
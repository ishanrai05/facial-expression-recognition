import os
from glob import glob
from tqdm import tqdm
import math
import argparse

# helper libraries
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.ion()   # interactive mode

# pytorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import get_dimensions
from get_data import get_data, face_array
from visualize import samples, compare, plot_confusion_matrix 
from model import initialize_model
from dataset import FRE2013
from train import train, validate

device = torch.device("cpu")


# '''
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--samples', type=bool, default=False, help='See sample images')
parser.add_argument('--view_data_counts', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
parser.add_argument('--model', type=str, default='densenet', help='resnet,vgg,densenet,inception')
parser.add_argument('--train', default=True, type=bool, help='train the model')

opt = parser.parse_args()


if opt.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''


emotion_type_dict = {
    0:"Angry",
    1:"Disgust",
    2:"Fear",
    3:"Happy",
    4:"Sad",
    5:"Surprise", 
    6:"Neutral"
}

root = os.path.join('./')

base_dir = os.path.join(root,'fer2013.csv')

df_train, df_test, df_val, df_inital = get_data(emotion_type_dict, base_dir)

width, height = get_dimensions(df_inital)

if opt.samples:
    faces = face_array(df_inital)
    samples(faces, df_inital, emotion_type_dict)

if opt.view_data_counts:
    compare(df_inital)

if opt.train:
    # resnet,vgg,densenet,inception
    model_name = opt.model
    num_classes = 7
    feature_extract = False
    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # norm_mean, norm_std = compute_img_mean_std(faces)

    norm_mean, norm_std = [0.5073955162068291], [0.2551289894150225]
    train_transform = transforms.Compose([
         transforms.RandomCrop(44),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std),
        ])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std)])

    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = FRE2013(df_train, transform=train_transform, width=width, height=height)
    train_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=4)
    # Same for the validation set:
    validation_set = FRE2013(df_val, transform=val_transform, width=width, height=height)
    val_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=4)

    # we use Adam optimizer, use cross entropy loss as our loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)


    import time
    since = time.time()
    epoch_num = opt.num_epochs
    best_val_acc = 0
    total_loss_val, total_acc_val = [],[]
    for epoch in tqdm((range(1, epoch_num+1))):
        print('\n')
        loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch, device)
        loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch, deviceh)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        if acc_val > best_val_acc:
                best_val_acc = acc_val
                print('*****************************************************')
                print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
                print('*****************************************************')
        print ('Time Taken: ',time.time()-since)
    fig = plt.figure(num = 2)
    fig1 = fig.add_subplot(2,1,1)
    fig2 = fig.add_subplot(2,1,2)
    fig1.plot(total_loss_train, label = 'training loss')
    fig1.plot(total_acc_train, label = 'training accuracy')
    fig2.plot(total_loss_val, label = 'validation loss')
    fig2.plot(total_acc_val, label = 'validation accuracy')
    plt.legend()
    plt.show()

    print ('\n Evaluating the model')
    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    plot_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised','Neutral']
    plot_confusion_matrix(confusion_mtx, plot_labels)
    # Generate a classification report
    report = classification_report(y_label, y_predict, target_names=plot_labels)
    print(report)
    label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    plt.bar(np.arange(7),label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import cv2
get_ipython().run_line_magic('reload_ext', 'tensorboard')


# In[2]:


def dataset_builder(size_img):

    #reading the train and test datasets and converting them to tensor
    #transforms.ToTensor will scale the images to 0-1 range
    #resize images using transforms.resize
    data_transform = transforms.Compose([
            transforms.Resize((size_img,size_img)),
            transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root='../data/train/img',
                                           transform=data_transform)
    train_dataloader = DataLoader(train_dataset,batch_size=len(train_dataset),
                                    shuffle=True,
                                    num_workers=4)    

    test_val_dataset=datasets.ImageFolder(root='../data/test/img',
                                           transform=data_transform)
    #splitting test dataset into test+val dataset such that each class in test and validation dataset has 
    # 500 and 300 images respectively. Also, Shuffling is done before the sets are created.
    dataset_size = len(test_val_dataset)   
    val_idx=[]
    test_idx=[]
    temp=[]
    for i in range(0, 8000, 800):
        temp=random.sample(range(i,i+800),800)    
        val_idx+=temp[0:300]
        test_idx+=temp[300:800]
    val_dataset=Subset(test_val_dataset, val_idx)
    test_dataset=Subset(test_val_dataset,test_idx)
    
    #normalizing the datasets. first calculate mean and std for each channel in each training image and take
    #the average of those values. Then normalize train,test,val datasets using these found values for training dataset.
    inputs, classes = next(iter(train_dataloader))
    #print("inputs",len(inputs))
    mean_r=0
    std_r=0
    mean_g=0
    std_g=0
    mean_b=0
    std_b=0
    for img in inputs:
        mean_r+=(img[0].mean()).item()
        std_r+=(img[0].std()).item()
        mean_g+=(img[1].mean()).item()
        std_g+=(img[1].std()).item()
        mean_b+=(img[2].mean()).item()
        std_b+=(img[2].std()).item()
    mean_R=mean_r/(len(inputs))
    mean_G=mean_g/(len(inputs))
    mean_B=mean_b/(len(inputs))
    std_R=std_r/(len(inputs))
    std_G=std_g/(len(inputs))
    std_B=std_b/(len(inputs))
    
    
    #now normalize the datasets using these values of mean and std for each channel
    data_transform_normalize = transforms.Compose([
            transforms.Resize((size_img,size_img)), transforms.ToTensor(),
        transforms.Normalize((mean_R,mean_G,mean_B),(std_R,std_G,std_B))])
    train_dataset_normalized = datasets.ImageFolder(root='../data/train/img',
                                           transform=data_transform_normalize)
        
    val_dataset.dataset.transform = transforms.Compose([transforms.Resize((size_img,size_img)), transforms.ToTensor(),
        transforms.Normalize((mean_R,mean_G,mean_B),(std_R,std_G,std_B))])
    
    test_dataset.dataset.transform = transforms.Compose([transforms.Resize((size_img,size_img)), transforms.ToTensor(),
        transforms.Normalize((mean_R,mean_G,mean_B),(std_R,std_G,std_B))])
    
    return train_dataset_normalized, val_dataset, test_dataset 
    
    


# In[3]:


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        
        #initialize first layer:conv->relu-->maxpooling
        self.conv1=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize second layer:conv->relu-->maxpooling
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize third layer:FC layer-->relu
        self.fc1=nn.Linear(in_features=16*5*5, out_features=120)
        self.relu3=nn.ReLU()
        
        #initialize fourth layer:FC layer-->relu
        self.fc2=nn.Linear(in_features=120, out_features=84)
        self.relu4=nn.ReLU()
        
        #initialize fifth layer:FC layer-->logSoftMax
        self.fc3=nn.Linear(in_features=84, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        #passing the input through first set of layers
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        #passing output from first layer through second set of layers
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        
        #passing output from second layer through third set of layers
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.relu3(x)
        
        #passing output from third layer through fourth set of layers
        x=self.fc2(x)
        x=self.relu4(x)
        
        #passing output from fourth layer through fifth set of layers
        x=self.fc3(x)
        output = self.logSoftmax(x)
        return output


# In[36]:


class LeNet5_BN(nn.Module):
    def __init__(self):
        super().__init__()
        
        #initialize first layer:conv->BatchNormalization-->relu-->maxpooling
        self.conv1=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.bn1=nn.BatchNorm2d(6)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize second layer:conv->BatchNormalization-->relu-->maxpooling
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn2=nn.BatchNorm2d(16)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize third layer:FC layer-->BatchNormalization-->relu
        self.fc1=nn.Linear(in_features=16*5*5, out_features=120)
        self.bn3=nn.BatchNorm1d(120)
        self.relu3=nn.ReLU()
        
        #initialize fourth layer:FC layer-->BatchNormalization-->relu
        self.fc2=nn.Linear(in_features=120, out_features=84)
        self.bn4=nn.BatchNorm1d(84)
        self.relu4=nn.ReLU()
        
        #initialize fifth layer:FC layer-->logSoftMax
        self.fc3=nn.Linear(in_features=84, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        #passing the input through first set of layers
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        #passing output from first layer through second set of layers
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        
        #passing output from second layer through third set of layers
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.bn3(x)
        x=self.relu3(x)
        
        #passing output from third layer through fourth set of layers
        x=self.fc2(x)
        x=self.bn4(x)
        x=self.relu4(x)
        
        #passing output from fourth layer through fifth set of layers
        x=self.fc3(x)
        output = self.logSoftmax(x)
        return output


# In[4]:


train_dataset, val_dataset, test_dataset=dataset_builder(32)
print("train size",len(train_dataset))
print("test size",len(test_dataset))
print("val size",len(val_dataset))


# In[30]:


def experiment(train_data,test_data,val_data,model_version,epochs,batchsize,learning_rate):
    
    # initializing the train, validation, and test data loaders with given batchsize
    trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=batchsize)
    valDataLoader = DataLoader(val_data, batch_size=batchsize)
    testDataLoader = DataLoader(test_data, batch_size=batchsize)
    
    # steps per epoch for train and validation set
    trainSteps = len(trainDataLoader.dataset) // batchsize
    valSteps = len(valDataLoader.dataset) // batchsize
    
    #initialize the model
    print("//...Initializing {} model...//".format(model_version))
    #model= LeNet5()
    model=model_version()
    
    #defining optimizer and cross-entropy loss function
    optimizer=Adam(model.parameters(),lr=learning_rate)
    criterion=nn.CrossEntropyLoss()
    
    #Learning rate will reduce by factor of 50% after every step size of 20epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    #empty dict to store losses and accuracy score
    myDict={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    
    #initialize training the model
    print("//....Training initiated...//")
    start=time.time()
    
    tb = SummaryWriter()
    for epo in range(0,epochs):
        model.train()
        trainLoss=0
        trainCorrect=0
        valLoss=0
        valCorrect=0
        for (x,y) in trainDataLoader:
            optimizer.zero_grad()
            ytrain_pred=model(x)
            loss=criterion(ytrain_pred,y)
            loss.backward()
            optimizer.step()
            trainLoss+=loss
            correct_train=(ytrain_pred.argmax(1)==y).type(torch.float).sum().item()
            trainCorrect=correct_train+trainCorrect
        with torch.no_grad():
            model.eval()
            for (x,y) in valDataLoader:
                yval_pred=model(x)
                loss=criterion(yval_pred,y)
                valLoss+=loss
                correct_val=(yval_pred.argmax(1)==y).type(torch.float).sum().item()
                valCorrect=correct_val+valCorrect
        scheduler.step()
        
        #calculating loss and accuracy for each epoch for both train and val sets
        trainLoss_avg=trainLoss/trainSteps
        valLoss_avg=valLoss/valSteps
        train_acc=trainCorrect/(len(trainDataLoader.dataset))
        val_acc=valCorrect/(len(valDataLoader.dataset))
        
        tb.add_scalar("train/Loss", trainLoss, epo)
        tb.add_scalar("train/Correct", trainCorrect, epo)
        tb.add_scalar("train/Accuracy", train_acc, epo)
        
        tb.add_scalar("val/Loss", valLoss, epo)
        tb.add_scalar("val/Correct", valCorrect, epo)
        tb.add_scalar("val/Accuracy", val_acc, epo)
        
        myDict["train_loss"].append(trainLoss_avg.cpu().detach().numpy())
        myDict["val_loss"].append(valLoss_avg.cpu().detach().numpy())
        myDict["train_acc"].append(train_acc)
        myDict["val_acc"].append(val_acc)
    
        print("//... epoch: {}/{}".format(epo + 1, epochs))
        print("Training loss: {:.4f}, Train accuracy: {:.2f}".format(trainLoss_avg, train_acc))
        print("Validation loss: {:.4f}, Validation accuracy: {:.2f}\n".format(valLoss_avg, val_acc))
    
    end=time.time()
    print("//...training the network took {:.2f} sec...//".format(end-start))
    
    #Evaluation of trained model on test set
    with torch.no_grad():
        model.eval()
        ytest_pred=[]
        ytrue=[]
        for (x,y) in testDataLoader:
            y_pred=model(x)
            ytest_pred.extend(y_pred.argmax(axis=1).numpy())
            ytrue.extend(y.numpy())
            
    #print("y_pred",ytest_pred)
    #print("y_true",ytrue)
    
    class_names = train_data.classes
    label_name = {'1': 'airplane', '2':'bird', '3':'car', '4':'cat', '5':'deer', '6':'dog', '7':'horse', '8':'monkey', '9':'ship', '10':'truck'}
    target_label=[]
    for i in class_names:
        target_label.append(label_name[i])
    #print("clas_names",class_names)
    #print("target_label",target_label)
    
    print(classification_report(ytrue,ytest_pred,target_names=target_label))
    print("//...Confusion Matrix...//")
    print(confusion_matrix(ytrue, ytest_pred, labels=[x for x in range(10)]))
    
    #plotting loss and acc
    plt.style.use("ggplot")
    plt.figure(2)
    plt.plot(myDict["train_loss"], label="train_loss")
    plt.plot(myDict["val_loss"], label="val_loss")
    plt.plot(myDict["train_acc"], label="train_acc")
    plt.plot(myDict["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy: {} model".format(type(model).__name__))
    plt.xlabel("# of Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    tb.close()


# In[39]:


def experiment_L2(train_data,test_data,val_data,model_version,epochs,batchsize,learning_rate):
    
    # initializing the train, validation, and test data loaders with given batchsize
    trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=batchsize)
    valDataLoader = DataLoader(val_data, batch_size=batchsize)
    testDataLoader = DataLoader(test_data, batch_size=batchsize)
    
    # steps per epoch for train and validation set
    trainSteps = len(trainDataLoader.dataset) // batchsize
    valSteps = len(valDataLoader.dataset) // batchsize
    
    #initialize the model
    print("//...Initializing {} model...//".format(model_version))
    #model= LeNet5()
    model=model_version()
    
    #defining optimizer and cross-entropy loss function. Also Adding L2 loss by specifying weight_decay
    optimizer=Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    criterion=nn.CrossEntropyLoss()
    
    #Learning rate will reduce by factor of 50% after every step size of 20epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    #empty dict to store losses and accuracy score
    myDict={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    
    #initialize training the model
    print("//....Training initiated...//")
    start=time.time()
    
    tb = SummaryWriter()
    for epo in range(0,epochs):
        model.train()
        trainLoss=0
        trainCorrect=0
        valLoss=0
        valCorrect=0
        for (x,y) in trainDataLoader:
            optimizer.zero_grad()
            ytrain_pred=model(x)
            loss=criterion(ytrain_pred,y)
            loss.backward()
            optimizer.step()
            trainLoss+=loss
            correct_train=(ytrain_pred.argmax(1)==y).type(torch.float).sum().item()
            trainCorrect=correct_train+trainCorrect
        with torch.no_grad():
            model.eval()
            for (x,y) in valDataLoader:
                yval_pred=model(x)
                loss=criterion(yval_pred,y)
                valLoss+=loss
                correct_val=(yval_pred.argmax(1)==y).type(torch.float).sum().item()
                valCorrect=correct_val+valCorrect
        scheduler.step()
        
        #calculating loss and accuracy for each epoch for both train and val sets
        trainLoss_avg=trainLoss/trainSteps
        valLoss_avg=valLoss/valSteps
        train_acc=trainCorrect/(len(trainDataLoader.dataset))
        val_acc=valCorrect/(len(valDataLoader.dataset))
        
        tb.add_scalar("train/Loss", trainLoss, epo)
        tb.add_scalar("train/Correct", trainCorrect, epo)
        tb.add_scalar("train/Accuracy", train_acc, epo)
        
        tb.add_scalar("val/Loss", valLoss, epo)
        tb.add_scalar("val/Correct", valCorrect, epo)
        tb.add_scalar("val/Accuracy", val_acc, epo)
        
        myDict["train_loss"].append(trainLoss_avg.cpu().detach().numpy())
        myDict["val_loss"].append(valLoss_avg.cpu().detach().numpy())
        myDict["train_acc"].append(train_acc)
        myDict["val_acc"].append(val_acc)
    
        print("//... epoch: {}/{}".format(epo + 1, epochs))
        print("Training loss: {:.4f}, Train accuracy: {:.2f}".format(trainLoss_avg, train_acc))
        print("Validation loss: {:.4f}, Validation accuracy: {:.2f}\n".format(valLoss_avg, val_acc))
    
    end=time.time()
    print("//...training the network took {:.2f} sec...//".format(end-start))
    
    #Evaluation of trained model on test set
    with torch.no_grad():
        model.eval()
        ytest_pred=[]
        ytrue=[]
        for (x,y) in testDataLoader:
            y_pred=model(x)
            ytest_pred.extend(y_pred.argmax(axis=1).numpy())
            ytrue.extend(y.numpy())
            
    #print("y_pred",ytest_pred)
    #print("y_true",ytrue)
    
    class_names = train_data.classes
    label_name = {'1': 'airplane', '2':'bird', '3':'car', '4':'cat', '5':'deer', '6':'dog', '7':'horse', '8':'monkey', '9':'ship', '10':'truck'}
    target_label=[]
    for i in class_names:
        target_label.append(label_name[i])
    #print("clas_names",class_names)
    #print("target_label",target_label)
    
    print(classification_report(ytrue,ytest_pred,target_names=target_label))
    print("//...Confusion Matrix...//")
    print(confusion_matrix(ytrue, ytest_pred, labels=[x for x in range(10)]))
    
    #plotting loss and acc
    plt.style.use("ggplot")
    plt.figure(2)
    plt.plot(myDict["train_loss"], label="train_loss")
    plt.plot(myDict["val_loss"], label="val_loss")
    plt.plot(myDict["train_acc"], label="train_acc")
    plt.plot(myDict["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy: {} model".format(type(model).__name__))
    plt.xlabel("# of Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    tb.close()


# In[34]:


experiment(train_dataset,test_dataset,val_dataset,LeNet5,100,128,0.001)


# In[38]:


experiment(train_dataset,test_dataset,val_dataset,LeNet5_BN,100,128,0.001)


# In[42]:


experiment_L2(train_dataset,test_dataset,val_dataset,LeNet5,100,128,0.001)


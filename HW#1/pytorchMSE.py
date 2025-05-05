import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tqdm import tqdm

import random

class labeledData(Dataset):
    def __init__(self,path):
        super().__init__()
        self.data = pd.read_csv(path)
        self.labels = self.data['label']
        self.data.drop(['label'],axis=1,inplace=True)
        self.dataTensor = torch.tensor(self.data.to_numpy(),dtype=torch.float32)
        self.labelTensor = torch.tensor(self.labels.to_numpy(),dtype=torch.float32)
        self.labelTensor = self.labelTensor.unsqueeze(1)

    def __len__(self):
        return self.dataTensor.shape[0]
    
    def __getitem__(self, idx):
        data = self.dataTensor[idx,:]
        label = self.labelTensor[idx,:]

        return data,label


class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(input,hidden)
        self.act = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden,output)


    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)

        return x
    

def train(model,dataloaderTrain,dataloaderVal,optimizer,lossFunc,epochs,device):

    lossTrainHolder = []
    lossValHolder = []
    bestVal = 100

    model.train()

    for epoch in tqdm(range(epochs)):

        lossTotal = 0

        for data,label in dataloaderTrain:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out = model(data)

            loss = lossFunc(out,label)

            loss.backward()
            optimizer.step()

            lossTotal += loss.item()    

        lossTrainHolder.append(lossTotal/len(dataloaderTrain))

        if epoch % 1 == 0:
            lossVal = test(model,dataloaderVal,lossFunc,device)
            lossValHolder.append(lossVal)
            if lossVal < bestVal:
                bestVal = lossVal
                torch.save(model.state_dict(),'/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/weights/weightsMSE.pth')

    return lossTrainHolder, lossValHolder


def test(model,dataloader,lossFunc,device):

    model.eval()

    lossTotal = 0

    with torch.no_grad():
        for data,label in dataloader:
            data = data.to(device)
            label = label.to(device)

            out = model(data)

            loss = lossFunc(out,label)

            lossTotal += loss.item()    

    return lossTotal/len(dataloader)

def testScore(model,dataloader,device):

    model.load_state_dict(torch.load('/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/weights/weightsMSE.pth'))

    model.eval()

    correct = 0
    with torch.no_grad():
        for data,label in dataloader:
            data = data.to(device)
            label = label.to(device).long()

            out = model(data)
            pred = out > 0.5

            correct += torch.sum(pred == label).item()
            
    accuracy = correct/len(dataloader.dataset)

    return accuracy

if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)


    device = torch.device('cpu')
    lr = 0.01
    hiddenNodes = 10
    batchSize = 8
    epochs = 3000

    lossFunc = nn.MSELoss()

    model = Model(2,hiddenNodes,1).to(device)

    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)

    name = 'center_surround'

    trainData = labeledData(f'/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/MyImplimentation/{name}_train.csv')
    valData = labeledData(f'/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/MyImplimentation/{name}_valid.csv')
    testData = labeledData(f'/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/MyImplimentation/{name}_test.csv')

    trainLoader = DataLoader(trainData,batchSize,True)
    valLoader = DataLoader(valData,batchSize,True)
    testLoader = DataLoader(testData,batchSize)

    trainHolder, valHolder = train(model,trainLoader,valLoader,optimizer,lossFunc,epochs,device)
    acc = testScore(model,testLoader,device)

    print(f"Best weights Accuracy: {acc}")

    plt.plot(range(epochs), trainHolder, label='Train')
    plt.plot(range(epochs), valHolder, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')  # change to 'Loss' or 'Accuracy' as appropriate
    plt.title('Training and Validation Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


    






import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class LinearLayer():
    def __init__(self,inputSize,outputSize):
        # Size in 64x3 3x10 -> 64,10
        self.weights = np.random.rand(inputSize,outputSize)
        self.bias = np.random.rand(1,outputSize)
        # if inputSize == 3:
        #     self.weights = np.array([[0.15,0.30],[0.20,0.35],[0.25,0.40]])
        #     self.bias = np.array([[0.35,0.35]])
        # else:
        #     self.weights = np.array([[0.45],[0.50]])
        #     self.bias = np.array([[0.6]])

    def forwardPass(self,x):
        self.input = x
        inter = np.dot(x,self.weights)
        output = inter + self.bias
        return output

    def backPropogation(self,prev):
        graph = Graph(True,True,None,True)
        # MAKE SURE TO CHECK ON THE SIZE OF THESE ARRAYS AND THAT THEY BECOME 3D WITH BATCHES 
        if prev.wandb:
            old = prev
            graph.b = np.sum(old.wPass,1).T
            repInp = np.concatenate([self.input for _ in range(old.wPass.shape[1])], axis=0)
            newInput = repInp.T
            graph.w = np.dot(newInput,old.wPass.T)
            graph.wPass = None
        else:
            graph.b = prev.value
            graph.w = np.concatenate([graph.b for _ in range(self.input.shape[1])], axis=0)
            # DOUBLE CHECK THIS MATH FOR MATRIX MULTIPLICATION
            graph.w = self.input.T * graph.w
            graph.wPass = np.concatenate([graph.b for _ in range(self.input.shape[1])], axis=0)
            graph.wPass = graph.wPass * self.weights


        return graph
    
class Sigmoid():
    def __init__(self):
        pass

    def forwardPass(self,x):
        self.input = x
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backPropogation(self,prev):
        # Make sure this also works with the linear layer
        sigPartial = self.out*(1-self.out)
        graph = prev
        if graph.wandb:
            graph.wPass = sigPartial.T * graph.wPass
        else:
            graph.value = sigPartial * graph.value
        return graph
    
class MSELoss():
    def __init__(self):
        pass

    def forwardPass(self,x,y):
        self.input = x
        self.label = y
        self.n = x.shape[0]
        inter = (x-y)*(x-y)
        mse = np.mean(inter)
        return mse
    
    def backPropogation(self):
        partial = 2*(self.input-self.label)/self.n
        result = Graph(True,True,partial)
        return result
    
class CELoss():
    def __init__(self):
        pass

    def forwardPass(self,x,y):
        self.input = x
        self.label = y
        return 1
    
    def backPropogation(self):
        output = np.zeros(self.input.shape)
        for i in range(self.label.shape[-1]):
            if self.label[0,i] == 1:
                output[0,i] = -1/self.input[0,i]
            else:
                output[0,i] = 1/(1-self.input[0,i])

        result = Graph(True,True,output)
        return result
    
class Softmax():
    def __init__(self):
        pass

    def forewardPass(self,x):
        self.input = x
        return np.exp(x)/np.sum(np.exp(x),1)
    
    def backPropogation(self,prev):
        dimTot = self.input.shape[1]
        dimLess = dimTot-1
        inpExp = np.exp(self.input)
        repInp = np.concatenate([inpExp for _ in range(dimLess)], axis=0).T
        dim0 = []
        for i in range(dimTot):
            dim1 = []
            for j in range(dimTot):
                if j != i:
                    dim1.append(inpExp[0,j])
            dim0.append(dim1)
        dim0 = np.array(dim0).T
        diag = np.dot(repInp,dim0)
        out = [diag[i,i] for i in range(diag.shape[0])]
        # partial = (inpExp*out)/(np.sum(inpExp,1)**2)
        partial = out/(np.sum(inpExp,1)**2)
        prev.value = prev.value * partial
        return prev

    
class Graph():
    def __init__(self,transfer=True,interWeight=False,value=None,wandb=False):
        self.transfer = transfer
        self.interWeight = interWeight
        self.value = value
        self.wandb = wandb
        if self.wandb:
            self.b = None
            self.w = None
            self.wPass = None

def updateStep(ups,layers,param):
    for i,_ in enumerate(layers):
        layer = layers[i]
        up = ups[i]
        # USE THE BELOW WHEN RUNNING WITH BATCHES
        # up.b = np.mean(up.b,axis=0)
        # up.w = np.mean(up.w,axis=0)

        layer.weights = layer.weights - param * up.w
        layer.bias = layer.bias - param * up.b

class Model():
    def __init__(self,input,hidden,output):
        self.layer1 = LinearLayer(input,hidden)
        self.sigmoid1 = Sigmoid()
        self.layer2 = LinearLayer(hidden,output)
        self.sigmoid2 = Sigmoid()
        self.loss = MSELoss()

    def forwardPass(self,x,labels):
        x = self.layer1.forwardPass(x)
        x = self.sigmoid1.forwardPass(x)
        x = self.layer2.forwardPass(x)
        x = self.sigmoid2.forwardPass(x)
        lossCalc = self.loss.forwardPass(x,labels)

        return lossCalc, x
    
    def backProp(self):
        der = self.loss.backPropogation()
        der = self.sigmoid2.backPropogation(der)
        up1 = self.layer2.backPropogation(der)
        der = self.sigmoid1.backPropogation(up1)
        up2 = self.layer1.backPropogation(der)

        return up1,up2
    
    def save(self,path):
        np.savez(path,w1=self.layer1.weights,b1=self.layer1.bias,w2=self.layer2.weights,b2=self.layer2.bias)

    def load(self,path):
        data = np.load(path)
        self.layer1.weights = data['w1']
        self.layer1.bias = data['b1']
        self.layer2.weights = data['w2']
        self.layer2.bias = data['b2']


def train(model,data,lr):

    loss = 0

    for i in range(data.shape[0]):

        x = data[i,1:]
        label = data[i,0]

        x = np.expand_dims(x,axis=0)
        label = np.array([[label]])

        # x = np.array([[0.05,0.10,0.15]])
        # label = np.array([[0.01]])

        lossTemp, _ = model.forwardPass(x,label)
        loss += lossTemp
        up1, up2 = model.backProp()
        layers = [model.layer2,model.layer1]
        ups = [up1,up2]
        updateStep(ups,layers,lr)

    loss = loss/data.shape[0]

    return loss

def test(model,data):
    
    loss = 0

    for i in range(data.shape[0]):

        x = data[i,1:]
        label = data[i,0]

        lossTemp, _ = model.forwardPass(x,label)

        loss += lossTemp
        
    loss = loss/data.shape[0]

    return loss

def testScore(model,data):

    correct = 0

    for i in range(data.shape[0]):

        x = data[i,1:]
        label = data[i,0]

        _, pred = model.forwardPass(x,label)

        predVal = (pred > 0.5)

        if predVal == label:
            correct += 1
        
    accuracy = correct/data.shape[0]

    return accuracy



def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    trainSet = pd.read_csv('/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/MyImplimentation/two_gaussians_train.csv')
    valSet = pd.read_csv('/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/MyImplimentation/two_gaussians_valid.csv')
    testSet = pd.read_csv('/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/MyImplimentation/two_gaussians_test.csv')

    trainNumpy = trainSet.to_numpy()
    valNumpy = valSet.to_numpy()
    testNumpy = testSet.to_numpy()

    model = Model(2,10,1)

    # XOR 2000,4hidden,0.1 (worked better with smaller learning rate (had 0.5 first))
    # Spiral, 2000, 10 hidden much better performance becasue was able to do a spiral, adjusted learning rate to 0.01 instead for smoothness
    # Center, 1000 8 hidden nodes, 0.05 worse performance than the other two models
    # Guassian, 2000, 10, 0.01


    epochs = 2000

    trainHolder = []
    valHolder = []

    for epoch in range(epochs):

        bestLoss = 100

        trainLoss = train(model,trainNumpy,0.01)
        trainHolder.append(trainLoss)
        lossVal = test(model, valNumpy)
        valHolder.append(lossVal)
        if lossVal < bestLoss:
            bestLoss = lossVal
            model.save('/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/weights/weights.npz')

    finalWeightsLoss = testScore(model,testNumpy)
    model.load('/Users/maxbengtsson/Desktop/Python/Spring 2025/CS 449/HW#1/weights/weights.npz')
    bestWeightsLoss = testScore(model,testNumpy)

    print(f"Final weights Accuracy: {finalWeightsLoss}")
    print(f"Best weights Accuracy: {bestWeightsLoss}")

    plt.plot(range(epochs), trainHolder, label='Train')
    plt.plot(range(epochs), valHolder, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')  # change to 'Loss' or 'Accuracy' as appropriate
    plt.title('Training and Validation Curve')
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()
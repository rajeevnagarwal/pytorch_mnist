import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(1,10,kernel_size=5)
		self.conv2 = nn.Conv2d(10,20,kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320,50)
		self.fc2 = nn.Linear(50,10)

	def forward(self,x):
		x  = F.relu(F.max_pool2d(self.conv1(x),2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		x = x.view(-1,320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x,training = self.training)
		x = self.fc2(x)
		x = F.log_softmax(x,dim=1)
		return x

def train(model,train_data,test_data,learning_rate,epochs,batch_size):
	optimizer = optim.SGD(model.parameters(),lr = learning_rate)
	for i in range(epochs):
		model.train()
		train_loss = 0
		for batch_idx,(data,target) in enumerate(train_data):
			optimizer.zero_grad()
			output = model(data)
			loss = F.nll_loss(output,target)
			train_loss = train_loss + loss
			loss.backward()
			optimizer.step()
		train_loss = train_loss/float(batch_size)
		print('Train Epoch : {} Loss : {:.6f}'.format(i,train_loss))
		torch.save(model.state_dict(),'model.ckpt')



def load_datasets():
	mnist_train = datasets.MNIST(root='./data',train=True,download=True,transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
	mnist_test = datasets.MNIST(root='./data',train=False,download=True,transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
	return mnist_train,mnist_test

if __name__=="__main__":
    train_data,test_data = load_datasets()
    model = Net()
    learning_rate = 0.1
    batch_size = 100
    epochs = 20
    train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_data,batch_size=batch_size,shuffle=True)
    train(model,train_loader,test_loader,learning_rate,epochs,batch_size)

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

def load_model(path):
	model = Net()
	model.load_state_dict(torch.load(path))
	return model

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

def test(model,test_data):
	n = len(test_data)
	model.eval()
	acc = 0
	wrong = 0
	with torch.no_grad():
		for i in range(n):
			img,target = test_data[i]
			img = img.unsqueeze_(0)
			out = model(img)
			pred = out.max(1,keepdim=True)[1][0][0]
			if(pred.eq(target)==0):
				wrong = wrong + 1
			else:
				acc = acc  + 1
		acc = float(acc)/n
	print(acc)


if __name__=="__main__":
	path = 'model.ckpt'
	model = load_model(path)
	train_data,test_data = load_datasets()
	test(model,test_data)

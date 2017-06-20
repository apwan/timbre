import numpy as np
print np.__file__
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import FloatTensor,IntTensor,LongTensor

class Net(nn.Module):
    def __init__(self,num_features):
        super(Net, self).__init__()
        self.l1=nn.Linear(num_features,128)
        self.l2=nn.Linear(128,16)
        self.l3=nn.Linear(16,3)
        #self.l4=nn.Linear(3,11)

    def forward(self,X):
        X=F.relu(self.l1(X))
        X=F.relu(self.l2(X))
        X=F.relu(self.l3(X))
        #X=F.relu(self.l4(X))
        return X

class Net2(nn.Module):
    def __init__(self,num_features):
        super(Net2, self).__init__()
        self.l1=nn.Linear(num_features,16)
        self.l2=nn.Linear(16,2)
        self.l3=nn.Linear(2,16)
        self.l4=nn.Linear(16,num_features)

    def forward(self,X):
        X=F.relu(self.l1(X))
        X=F.relu(self.l2(X))
        X=F.relu(self.l3(X))
        X=self.l4(X)
        return X

    def transform(self,X):
        X=F.relu(self.l1(X))
        X=F.relu(self.l2(X))
        return X

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		n_in,n_out=m.in_features,m.out_features
		std=np.sqrt(2.0/(n_in+n_out))
		print classname,n_in,n_out,std
		m.weight.data.normal_(0.0, std)
		m.bias.data.fill_(0.1)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class MLP(object):
	def __init__(self):
		return

	def fit(self,trainX,trainY):
		N,M=trainX.shape
		self.model=Net(num_features=M)
		self.model.apply(weights_init)
		#self.optimizer=optim.Adagrad(self.model.parameters(),lr=0.003,weight_decay=0.001)
		#self.optimizer=optim.SGD(self.model.parameters(),lr=0.001,momentum=0.95)
		self.optimizer=optim.Adam(self.model.parameters())
		self.model.train()
		#np.random.seed(0)
		criterion=nn.CrossEntropyLoss()

		for i in xrange(5000):
			batch_indices=np.random.randint(N,size=32)
			x=Variable(FloatTensor(trainX[batch_indices,:]),requires_grad=False)
			y=Variable(LongTensor(trainY[batch_indices]),requires_grad=False)
			#y=Variable(IntTensor(trainY[batch_indices]),requires_grad=False)
	
			self.optimizer.zero_grad()
			pred=self.model(x)
	
			loss=criterion(pred,y)
			if i % 100 == 0:
				print loss.data.numpy()
			loss.backward()
			
			self.optimizer.step()

	def predict(self,testX):
		x=Variable(FloatTensor(testX))
		self.model.eval()
		pred=self.model(x).data.numpy()
		print pred.shape
		return pred.argmax(axis=1)

class MLPRegressor(object):
	def __init__(self):
		return

	def fit(self,trainX):
		N,M=trainX.shape
		self.model=Net2(num_features=M)
		self.model.apply(weights_init)
		self.optimizer=optim.Adam(self.model.parameters())
		self.model.train()
		#np.random.seed(0)
		criterion=nn.MSELoss()

		for i in xrange(5000):
			batch_indices=np.random.randint(N,size=32)
			x=Variable(FloatTensor(trainX[batch_indices,:]),requires_grad=False)
	
			self.optimizer.zero_grad()
			reconstruct=self.model(x)
	
			loss=criterion(reconstruct,x)
			if i % 100 == 0:
				print loss.data.numpy()
			loss.backward()
			
			self.optimizer.step()

	def transform(self,testX):
		x=Variable(FloatTensor(testX))
		self.model.eval()
		transformed=self.model.transform(x).data.numpy()
		print transformed.shape
		return transformed


class LSTMRegressor(nn.Module):
	def __init__(self,input_size=1,hidden_size=2,num_layers=2):
		super(Net, self).__init__()
		self.lstm=torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
		self.linear=torch.nn.Linear(hidden_size*num_layers,1)

	def forward(self,packed):
		_,(h,_)=self.lstm(packed)
		h=h.permute(1,0,2).contiguous().view(batch_size,-1)
		pred=self.linear(h)
		return pred

	def init_weights(self):
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)	

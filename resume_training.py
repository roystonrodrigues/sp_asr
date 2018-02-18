import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import cv2
from numpy import linalg as LA
import os
import random
import torch.optim as optim



###################### Utility Function Definitions ##############################

def flatten(l):
	return list(itertools.chain.from_iterable(l))

def num_flat_features(x):
                
	    size = x.size()[1:]  
            num_features = 1
            for s in size:
             num_features *= s
            return num_features

##################################################################################

k_1="/home/SharedData/Royston/lipread_mp4"

dest="/media/royston/disk1/val/"




a1=os.listdir(dest)
a1.sort()


a_1=os.listdir(k_1)
a_1.sort()

############################# Validation Data ####################################

index=0;
dict_img_val={}
dict_img_val_label={}
for file in a1:
	a2=os.listdir(dest+file+"/pre_img/")
	a2.sort()
	for file_2 in a2:
		a3=os.listdir(dest+file+"/pre_img/"+file_2+"/val/")
		a3.sort()
		for file_3 in a3:
			i=dest+file+"/pre_img/"+file_2+"/val/"+file_3
			dict_img_val[index]=i


			dict_img_val_label[index]=a_1.index(file_2)

			index=index+1

print("Dictionary Image Validation Created")


dict_img_val = {k: dict_img_val[k] for k in dict_img_val.keys()}

dict_img_val_label={k: dict_img_val_label[k] for k in dict_img_val_label.keys()}

##############################################################################


dest="/media/royston/disk1/test/"
a1=os.listdir(dest)
a1.sort()

a_1=os.listdir(k_1)
a_1.sort()


############################# Test Data ####################################

index=0;
dict_img_test={}
dict_img_test_label={}
for file in a1:
	a2=os.listdir(dest+file+"/pre_img/")
	a2.sort()
	for file_2 in a2:
		a3=os.listdir(dest+file+"/pre_img/"+file_2+"/test/")
		a3.sort()
		for file_3 in a3:
			i=dest+file+"/pre_img/"+file_2+"/test/"+file_3
			dict_img_test[index]=i


			dict_img_test_label[index]=a_1.index(file_2)

			index=index+1

print("Dictionary Image Test Created")


dict_img_test = {k: dict_img_test[k] for k in dict_img_test.keys()}

dict_img_test_label={k: dict_img_test_label[k] for k in dict_img_test_label.keys()}

################################################################################



dest="/media/royston/disk1/train/"
a1=os.listdir(dest)
a1.sort()

a_1=os.listdir(k_1)
a_1.sort()


############################# Train Data ####################################

index=0;
dict_img_train={}
dict_img_train_label={}
for file in a1:
	a2=os.listdir(dest+file+"/pre_img/")
	a2.sort()
	for file_2 in a2:
		a3=os.listdir(dest+file+"/pre_img/"+file_2+"/train/")
		a3.sort()
		for file_3 in a3:
			i=dest+file+"/pre_img/"+file_2+"/train/"+file_3
			dict_img_train[index]=i


			dict_img_train_label[index]=a_1.index(file_2)

			index=index+1

print("Dictionary Image Test Created")


dict_img_train = {k: dict_img_train[k] for k in dict_img_train.keys()}

dict_img_train_label={k: dict_img_train_label[k] for k in dict_img_train_label.keys()}
#print("Length of Train Data")
#print(len(dict_img_train_label))

################################################################################






############################# Datasets Class Definitions ####################################

class bbc_lrw_val_img(Dataset):
    
	def __init__(self,transform=None):
		self.dummy=0

    	def __len__(self):
		return (25000) 

    	def __getitem__(self, idx):


		self.data = np.load(dict_img_val[idx])
		self.label = dict_img_val_label[idx]
		l=np.divide(self.data,255.0)
		#l=self.data
		l=np.float32(l)
		l=(l-0.5)/0.2
		
	      	#l[:,0]=(l[:,0]-0.485)/0.229
	      	#l[:,1]=(l[:,1]-0.456)/0.224
	      	#l[:,2]=(l[:,2]-0.406)/0.225
		

    		return l,self.label


class bbc_lrw_test_img(Dataset):
    
	def __init__(self,transform=None):
		self.dummy=0

    	def __len__(self):
		return (25000) 

    	def __getitem__(self, idx):


		self.data = np.load(dict_img_test[idx])
		self.label = dict_img_test_label[idx]
		l=np.divide(self.data,255.0)
		#l=self.data
		l=np.float32(l)
		l=(l-0.5)/0.2
		

    		return l,self.label

class bbc_lrw_train_img(Dataset):
    
	def __init__(self,transform=None):
		self.dummy=0

    	def __len__(self):
		return (489299) 

    	def __getitem__(self, idx):


		self.data = np.load(dict_img_train[idx])
		self.label = dict_img_train_label[idx]

		flip=random.randint(0,1)
		
		if flip==0:
			b1=self.data[:,:,:,::-1]	
		else:
			b1=self.data


		f_ind=random.randint(0,7)
		s_ind=random.randint(0,7)

		c1=b1[:,:,f_ind:f_ind+112,s_ind:s_ind+112]

		l=np.divide(c1,255.0)
		#l=self.data
		l=np.float32(l)
		l=(l-0.5)/0.2
		

    		return l,self.label



########################################################################################





########################################## MODEL DEFINITION ##########################################################################



class Net(nn.Module):

    def __init__(self,batch_sz):
        super(Net, self).__init__()

	self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
	self.input_bn1=nn.BatchNorm2d(30)	

        self.conv2 = nn.Conv2d(30, 60, 3, padding = 1)
	self.input_bn2=nn.BatchNorm2d(60)

        self.conv3 = nn.Conv2d(60, 120, 3, padding = 1)
	self.input_bn3=nn.BatchNorm2d(120)


        self.conv4 = nn.Conv2d(120, 240, 3, padding = 1)
	self.input_bn4=nn.BatchNorm2d(240)

	#self.conv5 = nn.Conv2d(256, 256, 3, padding = 1)
	#self.input_bn5=nn.BatchNorm2d(256)



	self.lstm1=nn.LSTM(30*56*56, 256, 2,batch_first=True,bidirectional=True)
	self.lstm2=nn.LSTM(60*28*28, 256, 2,batch_first=True,bidirectional=True)	# 2ed argument 256 for uni directional	
	self.lstm3=nn.LSTM(120*14*14, 256, 2,batch_first=True,bidirectional=True)
	self.lstm4=nn.LSTM(240*7*7,256, 2,batch_first=True,bidirectional=True)
	#self.lstm5=nn.LSTM(256,256, 2,batch_first=True,bidirectional=True)

	self.liner=nn.Linear(1024,500)
	
	#self.dense1_bn = nn.BatchNorm1d(1024) 

	self.softmax = nn.LogSoftmax(1)

	self.l1=torch.FloatTensor(np.zeros((batch_sz,29,30*56*56))).cuda()
	self.l2=torch.FloatTensor(np.zeros((batch_sz,29,60*28*28))).cuda()
	self.l3=torch.FloatTensor(np.zeros((batch_sz,29,120*14*14))).cuda()
	self.l4=torch.FloatTensor(np.zeros((batch_sz,29,240*7*7))).cuda()
	#self.l5=torch.FloatTensor(np.zeros((batch_sz,28,256))).cuda()

	
    	for name, param in self.lstm1.named_parameters():
  		if 'bias' in name:
     			nn.init.constant(param, 0.0)
  		elif 'weight' in name:
     			nn.init.xavier_normal(param)

    	for name, param in self.lstm2.named_parameters():
  		if 'bias' in name:
     			nn.init.constant(param, 0.0)
  		elif 'weight' in name:
     			nn.init.xavier_normal(param)

    	for name, param in self.lstm3.named_parameters():
  		if 'bias' in name:
     			nn.init.constant(param, 0.0)
  		elif 'weight' in name:
     			nn.init.xavier_normal(param)

    	for name, param in self.lstm4.named_parameters():
  		if 'bias' in name:
     			nn.init.constant(param, 0.0)
  		elif 'weight' in name:
     			nn.init.xavier_normal(param)
	'''
    	for name, param in self.lstm5.named_parameters():
  		if 'bias' in name:
     			nn.init.constant(param, 0.0)
  		elif 'weight' in name:
     			nn.init.xavier_normal(param)
	'''

    	
     			

	


    def forward(self, inputs,batch_sz):
		



		for i1 in range(0,batch_sz):

			cnn_feature_1=self.input_bn1(F.max_pool2d(F.relu(self.conv1(inputs[i1])),(2,2)))
			#print(inputs[i1])
		
			#t1a=F.max_pool2d(cnn_feature_1,(2,2))
			t1a=cnn_feature_1

			self.l1[i1]=t1a.view(-1,num_flat_features(t1a)).data

			cnn_feature_2=self.input_bn2(F.max_pool2d(F.relu(self.conv2(cnn_feature_1)),(2,2)))

			#t1b=F.max_pool2d(cnn_feature_2,(2,2))

			t1b=cnn_feature_2

			self.l2[i1]=t1b.view(-1,num_flat_features(t1b)).data

			cnn_feature_3=self.input_bn3(F.max_pool2d(F.relu(self.conv3(cnn_feature_2)),(2,2)))

			#t1c=F.max_pool2d(cnn_feature_3,(2,2))

			t1c=cnn_feature_3

			

			self.l3[i1]=t1c.view(-1,num_flat_features(t1c)).data
		
			cnn_feature_4=self.input_bn4(F.max_pool2d(F.relu(self.conv4(cnn_feature_3)),(2,2)))
			#print(cnn_feature_4)

			self.l4[i1]=cnn_feature_4.view(-1,num_flat_features(cnn_feature_4)).data

			#cnn_feature_5=self.input_bn5(F.max_pool2d(F.relu(self.conv5(cnn_feature_4)),(7,7)))

			#self.l5[i1]=cnn_feature_5.view(-1,num_flat_features(cnn_feature_5)).data

		
		packed_input_1 = pack_padded_sequence(Variable(self.l1), np.array([29]*batch_sz,dtype=int),batch_first=True)
		packed_input_2 = pack_padded_sequence(Variable(self.l2), np.array([29]*batch_sz,dtype=int),batch_first=True)
		packed_input_3 = pack_padded_sequence(Variable(self.l3), np.array([29]*batch_sz,dtype=int),batch_first=True)
		packed_input_4 = pack_padded_sequence(Variable(self.l4), np.array([29]*batch_sz,dtype=int),batch_first=True)
		#packed_input_5 = pack_padded_sequence(Variable(self.l5), np.array([28]*batch_sz,dtype=int),batch_first=True)
	

		packed_output_a, (hta,cta) = self.lstm1(packed_input_1)
		packed_output_b, (htb,ctb) = self.lstm2(packed_input_2)
		packed_output_c, (htc,ctc) = self.lstm3(packed_input_3)
		packed_output_d, (htd,ctd) = self.lstm4(packed_input_4)
		#packed_output_e, (hte,cte) = self.lstm5(packed_input_5)

		#print(hta)
	
		f_feature=torch.cat((hta[3]+hta[2], htb[2]+htb[3], htc[2]+htc[3],htd[2]+htd[3]),1)

		#print(f_feature)
		
		predictions=self.softmax(self.liner((f_feature)))

		

		return predictions


######################################################################################################################################

net=Net(8)
net=net.cuda()
net.load_state_dict(torch.load('image_model.pth'))
sumloss=0
transformed_dataset_train = bbc_lrw_train_img()
transformed_dataset_test = bbc_lrw_test_img()
transformed_dataset_val = bbc_lrw_val_img()

trainloader = torch.utils.data.DataLoader(transformed_dataset_train, batch_size=8,shuffle=True, num_workers=8,drop_last=True)
print("Data_loader for Train is Ready ")
valloader = torch.utils.data.DataLoader(transformed_dataset_val, batch_size=8,shuffle=True, num_workers=8,drop_last=True)
print("Data_loader for Validation is Ready ")
testloader = torch.utils.data.DataLoader(transformed_dataset_test, batch_size=8,shuffle=True, num_workers=8,drop_last=True)
print("Data_loader for test is Ready ")



#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
#optimizer = optim.Adam(net.parameters(),lr =0.001) #lr=0.00002
optimizer = optim.SGD(net.parameters(), lr =0.001, momentum=0.9)
optimizer.load_state_dict(torch.load('./image_optimizer.pth'))
acc_sum=0
k_z=160000
while(1):
	for i_a, data_a in enumerate(trainloader, 0):
		#print(i_a)


		net.train(True)
		inputs,labels=data_a

		outputs = net(Variable(inputs.cuda()),8)

		#print(outputs.data.max(1)[1])

		labels=Variable(labels.cuda())

		loss = criterion(outputs, labels)
		sumloss=sumloss+loss.data[0]
		loss.backward()
		##torch.nn.utils.clip_grad_norm(net.parameters(), 400)

		predict1 = outputs.data.max(1)[1]
		acc = predict1.eq(labels.data).cpu().sum()	

		acc_sum += acc


		optimizer.step()
		optimizer.zero_grad()
			#print(i_a/16)
		
			#print("Outputs")
			#print(outputs.data)
		
		#print("Max")
		#print(outputs.data.max(1)[1])
		
		#print("Labels")
		#print(labels.data)
		
		acc_sum_b=0

		if(k_z%10000== 0):
		


			np.savetxt('Image_Train_acc.txt', np.array([acc_sum]), fmt='%f')
			np.savetxt('Image_Train_Iter_batchsz_64.txt', np.array([k_z]), fmt='%f')
			acc_sum_b=0

			torch.save(net.state_dict(), './image_model.pth')
			torch.save(optimizer.state_dict(), './image_optimizer.pth')
			
			
			for i_b, data_b in enumerate(testloader, 0):
				net.train(False)
				inputs_b,labels_b=data_b
				labels_b=Variable(labels_b.cuda())
				outputs_b = net(Variable(inputs_b.cuda(),volatile=True),8)
				predict1_b= outputs_b.data.max(1)[1]
				acc_b = predict1_b.eq(labels_b.data).cpu().sum()	
				acc_sum_b += acc_b
			print('Evaluation (Total Test ) :: '+str(25000)+' Model got Right :: '+str(acc_sum_b) +' Accuracy Percentage :: '+str(acc_sum_b/25000.0) )
			
			for param_group in optimizer.param_groups:
        			param_group['lr'] = param_group['lr']/1.05
			
	
		k_z=k_z+1;
	

		print(':: Iteration Number :: ',k_z)
		print(':: Loss on Current Training Batch (8) :: ',sumloss)


		
		sumloss=0
		acc_sum=0


	
	





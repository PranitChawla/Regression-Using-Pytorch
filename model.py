import torch 
from math import pi
from math import sqrt
import numpy as np
from torch.autograd import Variable
import torch.optim as optim 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import sin
import imageio

class Model (nn.Module):
	def __init__ (self,n_feature,n_hidden,n_output):
		super(Model, self).__init__()
		self.hidden=nn.Linear(n_feature,n_hidden)
		self.predict=nn.Linear(n_hidden,n_output)
	def forward(self,x):
		x=self.hidden(x)
		x=F.relu(x)
		x=self.predict(x)
		return x

num_input=1
num_hidden=20
num_output=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
data_mean=0
data_std=1
y=x*x+0.2*torch.rand(x.size())
print (y)
print (x)
x=Variable(x)
x_gpu=x.cuda()
y_gpu=y.cuda()
y=Variable(y)
plt.figure(figsize=(10,4))
my_images = []
fig, ax = plt.subplots(figsize=(12,7))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
Net=Model(num_input,num_hidden,num_output).cuda()
optimizer=optim.SGD(Net.parameters(),lr=0.02)
loss_func=nn.MSELoss()
for step in range (2000):
	output=Net(x_gpu)
	optimizer.zero_grad()
	loss=loss_func(output,y_gpu)
	loss.backward()
	print (loss)
	optimizer.step()
	y_output=Net(x_gpu)
	if step%10==0:
		plt.cla()
		ax.set_title('Regression Analysis', fontsize=35)
		ax.set_xlabel('Independent variable', fontsize=24)
		ax.set_ylabel('Dependent variable', fontsize=24)
		ax.set_xlim(-1.05, 1.5)
		ax.set_ylim(-0.25, 1.25)
		ax.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
		ax.plot(x.data.numpy(), output.cpu().data.numpy(), 'g-', lw=3)
		ax.text(1.0, 0.1, 'Step = %d' % step, fontdict={'size': 24, 'color':  'red'})
		ax.text(1.0, 0, 'Loss = %.4f' % loss.cpu().data.numpy(),fontdict={'size': 24, 'color':  'red'})
		fig.canvas.draw()       # draw the canvas, cache the renderer
		image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
		image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		my_images.append(image)
		imageio.mimsave('./curve_4.gif', my_images, fps=10)
# plt.plot(x.data.numpy(), y_output.cpu().data.numpy(), color = "blue")
# plt.show()

	        

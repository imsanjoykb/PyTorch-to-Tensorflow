#!/usr/bin/env python
# coding: utf-8

# # Open Nueral Network Exchange [ONNX]

# ###Installing ONNX and other required libraries

# In[ ]:


get_ipython().system('pip install onnx')


# In[ ]:


get_ipython().system('pip install tensorflow-addons')
get_ipython().system('git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && pip install -e .')


# ### Restart Runtime before continuing

# In[ ]:


get_ipython().system('pip install torchvision')


# ###Import required libraries and classes

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import onnx
from onnx_tf.backend import prepare


# ### Define the model

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ### Create the train and test methods

# In[ ]:


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch,  loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


# ### Download the datasets, normalize them and train the model

# In[ ]:


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True)


torch.manual_seed(1)
device = torch.device("cuda")

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
for epoch in range(21):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)


# ### Save the Pytorch model

# In[ ]:


torch.save(model.state_dict(), 'mnist.pth')


# ### Load the saved Pytorch model and export it as an ONNX file

# In[ ]:


trained_model = Net()
trained_model.load_state_dict(torch.load('mnist.pth'))

dummy_input = Variable(torch.randn(1, 1, 28, 28)) 
torch.onnx.export(trained_model, dummy_input, "mnist.onnx")


# ### Load the ONNX file and import it into Tensorflow

# In[ ]:


# Load the ONNX file
model = onnx.load('mnist.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)


# ### Run and test the Tensorflow model

# In[ ]:


import numpy as np
from IPython.display import display
from PIL import Image
print('Image 1:')
img = Image.open('/content/img1.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))
print('------------------------------------------------------------------------------')
print('Image 2:')
img = Image.open('/content/img2.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))


# In[ ]:


tf_rep.export_graph('mnist.pb')


# In[ ]:





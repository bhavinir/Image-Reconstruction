
# Importing the libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils

batchSize = 128
imageSize = 28

# Creating the transformations
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 

# Loading the dataset
dataset = datasets.MNIST('~/data/mnist/', download = True, train = True, transform = transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 4) 

class CAE(nn.Module):

    def __init__(self): 
        super(CAE, self).__init__() 
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 16, 5, 1, 1), #16,26,26
            nn.ReLU(True),
            nn.MaxPool2d(2), #16,13,13
            nn.Conv2d(16, 32, 4, 1, 1), #32,12,12            
            nn.ReLU(True),
            nn.MaxPool2d(2), #32,6,6
            nn.Conv2d(32, 64, 3, 1, 0), #64,4,4            
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 1, 0), #32,7,7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), #16,14,14       
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), #1,28,28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)        
        return x
    
cae = CAE()
criterion = nn.MSELoss() 
optimizer = optim.Adam(cae.parameters(), lr = 0.0001,  weight_decay = 0.002)

#Training
for epoch in range(15):
    for i, data in enumerate(dataloader):       
        images, _ = data 
        image = Variable(images)
        optimizer.zero_grad()
        outputs = cae(image)
        loss = criterion(outputs, image)
        loss.backward()
        optimizer.step()
        
    print('[%d/%d] Loss: %.4f' % (epoch+1, 15, loss[0]))  
    vutils.save_image(images, '%s/original_samples.png' % "./results", normalize = True) 
    vutils.save_image(outputs.data, '%s/Reconstructed_samples_epoch_%03d.png' % ("./results", epoch+1), normalize = True)
     
    
    
#Testing
test_data  = datasets.MNIST('~/data/mnist/', download = True, train = False, transform = transform)        
test_dataloader = torch.utils.data.DataLoader(test_data) 
        
for i, data in enumerate(test_dataloader,0):       
    images, _ = data 
    image = Variable(images)
    optimizer.zero_grad()
    outputs = cae(image)
    test_loss = criterion(outputs, image)
    
print('[%d/%d] Test_Loss: %.4f' % (i+1, len(test_dataloader), test_loss[0]))
vutils.save_image(images, '%s/original_test_sample.png' % "./results", normalize = True) 
vutils.save_image(outputs.data, '%s/Reconstructed_test_sample.png' % ("./results"), normalize = True)
     
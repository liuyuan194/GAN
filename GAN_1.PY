import torch
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os

if not os.path.exists('./img_self'):
    os.mkdir('./img_self')



Batch_size = 100
z_dimension = 100 
epochs = 50

img_tansform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])


mnist = datasets.MNIST('./image',train=True,download=True,transform=img_tansform)
dataloader = torch.utils.data.DataLoader(mnist,batch_size=Batch_size,shuffle=True)
#print(dataloader)

class Generator(nn.Module):
    def __init__(self):

        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,784),
            nn.BatchNorm1d(784,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(784,784),
            nn.Tanh(),
        )
    def forward(self,x):
        gen  = self.gen(x)
        #gen  = gen.view(100,1,28,28)
        return gen

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        dis = self.dis(x)
        #dis = dis.view(784,-1)
        return dis

#loss function
criterion = torch.nn.BCELoss()

D = Discriminator()
G = Generator()


optimizer_D = torch.optim.Adam(D.parameters(),lr=0.0003,betas=(0.5,0.999))
optimizer_G = torch.optim.Adam(G.parameters(),lr=0.0003,betas=(0.5,0.999))


for epoch in range(epochs):
    for i, (img,_) in enumerate(dataloader):
        num_img = img.size(0)
        # ========================================================================TRAING D
        img = img.view(num_img, -1)
     
        real_img = Variable(img)
    
        real_label = Variable(torch.ones(num_img))
        fake_label = Variable(torch.zeros(num_img))
 
       
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  
 
        
        
        z = Variable(torch.randn(num_img, z_dimension))
        fake_img = G(z)
        
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  
 
        
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
 
        # ===================================================================== TRAING G
    
        z = Variable(torch.randn(num_img, z_dimension))
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        # BACK
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
 
        if (i + 1) % 300 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f},D real: {:.6f}, D fake: {:.6f}'.format(
                      epoch, epochs, d_loss.item(), g_loss.item(), 
                      real_scores.data.mean(), fake_scores.data.mean()))
        c= fake_img.view(-1,1,28,28)
        #print(c.shape)
        batches_done = epoch * len(dataloader) + i
        if batches_done % 400 == 0:
            save_image(c.data[:128], "img_self/%d.png" % batches_done, nrow=16, normalize=True)


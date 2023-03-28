
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.x_dim=x_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var,z

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var,x_dim=128):
    MSE = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # print("MSE:"+str(MSE))
    # print("KLD:"+str(KLD))
    return MSE + KLD

class AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(AE, self).__init__()
        self.x_dim=x_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h) # mu, log_var
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        z = self.encoder(x.view(-1, self.x_dim))
        return self.decoder(z),z
def loss_function_AE(recon_x, x,x_dim=128):
    MSE = F.mse_loss(recon_x, x.view(-1, x_dim), reduction='sum')
    return MSE

class decoder(nn.Module):
    def __init__(self,in_dim,h_dim1,h_dim2,out_dim):
        super(decoder,self).__init__()

        self.decode=nn.Sequential(
            nn.Linear(in_dim,h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1,h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2,out_dim)
        )
    def forward(self,z):
        return self.decode(z)

    

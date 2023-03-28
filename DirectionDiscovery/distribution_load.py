
import os,sys
from textwrap import indent
o_path = os.getcwd()
import torch
sys.path.append(o_path)

from DirectionDiscovery.distribution import Uniform_to_latent_Network

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model=Uniform_to_latent_Network().to(device=device)

model.load_state_dict(torch.load('./DirectionDiscovery/train/trained_model/1.pth'))
model.eval()
for i in range(1):
    in_data=((torch.rand((200,2))*160)-80).to(device=device)
    output=model(in_data)
    print(output)
    print(torch.mean(output,dim=0),torch.std(output,dim=0))


# 模型加载好了
# 加载GAN

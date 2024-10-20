import torch 
import torch.nn as nn 
      
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Preduction_MLP(nn.Module):
    def __init__(self):
        super(Preduction_MLP, self).__init__()
        self.linear = nn.Linear(28*28, 10)
        
        
    def forward(self, x):
        out = self.linear(x)
        return out

model = Preduction_MLP().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
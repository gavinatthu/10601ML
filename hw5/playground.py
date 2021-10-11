import torch

a = torch.tensor(())
for i in range(3):
    # if i = torch.tensor(1)
    # it cannot be cat, since it has 
    # zero dimention.
    # Also use .float() to make sure that they 
    # are in the same dtype
    i = torch.rand((10,240,360))
    a = torch.cat((a, i), 0)
print(a.shape)
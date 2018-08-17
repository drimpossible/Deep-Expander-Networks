import torch
#from expandergraphlayersparse import expanderLinear as Lin
from expandergraphlayer import expanderLinear as Lin
from expandergraphlayer import MulExpander as mul
from torch.autograd import Variable, Function
from torch.autograd import gradcheck

mask = torch.zeros(50,20)
for i in range(20):
    x = torch.randperm(50)
    for j in range(5):
        mask[x[j]][i] = 1
mask =  mask.double()
mask = Variable(mask, requires_grad=False)
#linear = Linear.apply
linear = Lin(mask)
# gradchek takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(50,20).double(), requires_grad=True))
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)

mask = torch.zeros(25, 32, 1, 1)
for i in range(25):
    x = torch.randperm(32)
    for j in range(20):
        mask[i][x[j]][0][0] = 1

mask = mask.repeat(1, 1, 5, 5)
mask =  mask.double()
mask = Variable(mask, requires_grad=False)
#print(mask.size())

c = mul(mask)

inputvar = (Variable((torch.randn(5, 32, 5, 5).double()), requires_grad=True),)
#print(inputvar.size())
test = gradcheck(c, inputvar, eps=1e-6, atol=1e-4)
print(test)

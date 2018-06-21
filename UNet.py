
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from collections import OrderedDict


# In[2]:


from torch.autograd import Variable
from torch.nn import init #used for initializations


# In[3]:


import numpy as np


# In[12]:


def conv3x3(in_channels,out_channels,stride=1,padding=1,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,bias=bias,groups=groups)

def upconv2x2(in_channels,out_channels,mode='transpose'):
    if mode=='transpose':
        return nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='Bilinear',scale_factor=2),conv1x1(in_channels,out_channels))
    
def conv1x1(in_channels,out_channels,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,groups=groups,stride=1)




# In[13]:


class Downconv(nn.Module):
    def __init__(self,in_channels,out_channels,pooling=True):
        
        super(Downconv,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.pooling=pooling
    
        self.conv1=conv3x3(self.in_channels,self.out_channels)
        self.conv2=conv3x3(self.out_channels,self.out_channels)
    
        if self.pooling:
            self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
            
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        before_pool=x
        if self.pooling:
            x=self.pool(x)
        return x,before_pool    
        


# In[14]:


class UpConv(nn.Module):
    
    def __init__(self,in_channels,out_channels,merge_mode='concat',up_mode='transpose'):
        super(UpConv,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.merge_mode=merge_mode
        self.up_mode=up_mode
        
        self.upconv=upconv2x2(self.in_channels,self.out_channels,mode=self.up_mode)
        
        if self.merge_mode=='concat':
            self.conv1=conv3x3(2*self.out_channels,out_channels)
        else:
            self.conv1=conv3x3(self.out_channels,self.out_channels)
        self.conv2=conv3x3(self.out_channels,self.out_channels)    
        
    def forward(self,from_down,from_up):
        #from up: tensor from decoder path
        #from_down tensor to concatenate which is the before_pool tensor in encoder arch
        from_up=self.upconv(from_up)
        if self.merge_mode=='concat':
            x=torch.cat((from_up,from_down),1) #1 gives the dimension to concat along
        else:
            x=from_up+from_down
            
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        return x
    
    


# In[15]:


range(5)


# In[18]:


class Unet(nn.Module):
    def __init__(self,num_classes,in_channels=3,depth=5,start_filts=64,up_mode='transpose',merge_mode='concat'):
        
        super(Unet,self).__init__()
        if up_mode in ('transpose','upsample'):
            self.up_mode=up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
            
        
        if merge_mode in ('concat','add'):
            self.merge_mode=merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "merging. Only \"concat\" and "
                             "\"add\" are allowed.".format(merge_mode))
        
        if self.merge_mode=='add' and self.up_mode=='upsample':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")
            
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.depth=depth
        self.start_filts=start_filts
        
        self.down_convs=[]
        self.up_convs=[]
        
        #create the encoder pathway and add to a list
        for i in range(depth):
            ins=self.in_channels if i==0 else outs
            outs=self.start_filts*(2**i)
            pooling=True if i<(depth-1) else False

            downconv=Downconv(ins,outs,pooling=pooling) 
            self.down_convs.append(downconv) #list of modules in the down path
            
        
        #create the decoder pathway and add to a list
        for i in range(depth-1):
            ins=outs
            outs=ins//2
            up_conv=UpConv(ins,outs,up_mode=up_mode,merge_mode=merge_mode)
            self.up_convs.append(up_conv) #list of modules in the up path 
            
        self.conv_final=conv1x1(outs,num_classes)
            
        #add the list of modules to the current module
        self.down_convs=nn.ModuleList(self.down_convs)
        self.up_convs=nn.ModuleList(self.up_convs)
            
        self.reset_params()
            
    #def weight_init(m):
        #if isinstance(m,nn.Conv2d):
            #init.xavier_normal(m.weight.double())
            #init.constant(m.bias,0)
            
    def reset_params(self):
        for i,m in enumerate(self.modules()):#self.modules returns an overall iterator over the modules of the net
            if isinstance(m,nn.Conv2d):
                init.xavier_normal(m.weight).double()
                init.constant(m.bias,0)
            
    def forward(self,x):
        encoder_outs=[]
        
        for i,module in enumerate(self.down_convs):
            x,before_pool=module(x)
            encoder_outs.append(before_pool)
            
        for i,module in enumerate(self.up_convs): 
            before_pool=encoder_outs[-(i+2)]
            x=module(before_pool,x)
        
        #no softmax but 1x1 conv is used for generating labels
        x=self.conv_final(x)
        return x
    


# In[19]:



if __name__=='__main__':
    
    #testing on a random input
    model=Unet(3,depth=5,merge_mode='concat')
    x=Variable(torch.FloatTensor(np.random.random((1,3,320,320))))
    out = model(x)
    loss=torch.sum(out)
    loss.backward()
                
    


# In[21]:


print loss


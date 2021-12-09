"""
NN architecture built after X2CT-GAN paper. 

Fixed batch_size at 1 and input image sizes at 128x128, resulting in fixed `feature_list = [180, 168, 144, 96, 32]` for Connection_B outputs. And also the magic number `744 * bs` as bridge between encoder and decoder for Connection_A. The output of the encoder can not be smoothly reshaped into a 3d cube array and thus has to be rounded to fit 1xNx4x4x4. 


"""


import math
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor
from torchvision import models
from collections import OrderedDict
from collections import namedtuple

# DenseLayer and Denseblock taken from Pytorch Densenet architecture
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Basic2d(nn.Module):
    """
    'Basic2d' Convolution Block. 

    Conv2d + Instance Norm + ReLU 

    Parameters
    ----------
    in_features: int
        Number of layer input features.
    out_features: int
        Number of layer output features.
    """
    def __init__(self, in_features, out_features):
        
        super(_Basic2d, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.relu(self.in1(self.conv1(x)))
        return out

class _Basic3d(nn.Module):
    """
    'Basic3d' Convolution Block. 

    Conv3d + Instance Norm + ReLU 

    Parameters
    ----------
    in_features : int

        Number of layer input features.

    out_features : int

        Number of layer output features.

    n : int, default = 1

        Number of consecutive Basic3d blocks. Might be used to increase features and layer depth. 
    """
    def __init__(self, in_features, out_features, n=1):
        super(_Basic3d, self).__init__()
        
        self.basic3d_block = nn.Sequential(OrderedDict([]))
        for i in range(n-1):
            block1 = nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1)
            block2 = nn.InstanceNorm3d(in_features)
            block3 = nn.ReLU()
            self.basic3d_block.add_module('conv3d%d' % (i + 1), block1)
            self.basic3d_block.add_module('in%d' % (i + 1), block2)
            self.basic3d_block.add_module('relu%d' % (i + 1), block3)

        block1 = nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        block2 = nn.InstanceNorm3d(out_features)
        block3 = nn.ReLU()
        self.basic3d_block.add_module('conv3d%d' % (n), block1)
        self.basic3d_block.add_module('in%d' % (n), block2)
        self.basic3d_block.add_module('relu%d' % (n), block3)


    def forward(self,x):
        out = self.basic3d_block(x)
        return out


class _Down(nn.Module):
    """
    'Down' Downsampling Block. 

    Instance Norm + ReLU + Conv2d 

    Parameters
    ----------
    in_features: int
        Number of layer input features.
    
    """
    def __init__(self, in_features):
        super(_Down, self).__init__()
        self.in1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_features, in_features, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        out = self.conv2d(self.relu(self.in1(x)))
        return out

class _Compress(nn.Module):
    #half channels
    def __init__(self, in_features):
        super(_Compress, self).__init__()
        self.in1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_features, int(in_features*0.5), kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        out = self.conv2d(self.relu(self.in1(x)))
        return out

class _Dense(nn.Module):
    """
    'Dense' Module. 

    'Down' + 'Dense Block' + 'Compress' 

    Parameters
    ----------
    num_features: int
        Number of input features.
    bn_size: int
        Batch size. 
    num_layers: int
        default = 6
        Number of dense layers used.
    growth_rate: int
        default = 32
        Growth rate inbetween layers  
    drop_rate: float
        default = 0.0
        Drop rate. Possibly unnecessary
    """
    
    def __init__(self, num_features, bn_size, growth_rate, num_layers=6,  drop_rate=0.0 ):
        
        super(_Dense, self).__init__()
        num_features = int(num_features)
        self.down = _Down(num_features)
        self.dense_block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.compress = _Compress(num_features)
    
    def forward(self, x):
        # print("Dense: ")
        # print(x.shape)
        out = self.down(x)
        # print(out.shape)
        out = self.dense_block(out)
        # print(out.shape)
        out = self.compress(out)
        return out




class ConnectionA(nn.Module):
    """
    'ConnectionA' bridge. 

    FC -> Dropout -> ReLU -> Reshape to 4x4x4 cube.

    Image size of input has to be rounded off first for it to fit the cube. Possible information loss here. 

    Parameters
    ----------
    size: int
        Size of flattened input tensor.
    output_dim: int
        Width of output cube.  
    """

    def __init__(self, size, output_size, output_shape):
        super(ConnectionA, self).__init__()

        # output_size = size - size%int(output_dim**3)
        # c = output_size/int(output_dim**3)

        # self.out_shape = [1, int(c), output_dim, output_dim, output_dim]
        self.output_shape = output_shape
        self.fc = nn.Linear(size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        out = self.fc(x)
        out = self.relu(F.dropout(out, 0.3))
        out = out.view(self.output_shape)
        
        return out

class ConnectionB(nn.Module):
    """
    'ConnectionB' bridge.

    Basic2d -> Expand(stack to force dimensions) -> Basic3d

    Parametes
    ---------
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.
    """

    def __init__(self, in_features, out_features):
        super(ConnectionB, self).__init__()
        
        self.basic2d = _Basic2d(in_features, out_features)
        self.basic3d = _Basic3d(out_features, out_features)

    def forward(self, x, size):
        """
        Parameters: 
        -----------
        x: tensor
            2D input tensor.
        size: int
            Size of b_list input.
        """
        out = self.basic2d(x)
        # stack tensor to force 3d shape
        stacked_out = out[:, :, None].repeat(1, 1, size, 1, 1) 
        stacked_out = self.basic3d(stacked_out)

        return stacked_out
        
class ConnectionC(nn.Module):
    """
    'ConnectionC' bridge. 

    Permutes input cubes to the same direction and calculates average.
    """
    def __init__(self):
        super(ConnectionC, self).__init__()
    
    def forward(self, x, y):
        """
        Parameters
        ----------
        x: 5d Tensor
        y: 5d Tensor
        """
        #permute: roty 90d, rotx -90d
        #assuming x comes from S direction and y comes from R direction
        x = x.transpose(3,4).flip(3) 
        x = x.transpose(2,4).flip(4)

        out = torch.add(x,y)
        out = torch.div(out, 2)
        return out
    
class _Up(nn.Module):
    """
    'Up' Block

    Deconv3d + Instance Norm + ReLU

    Doubles feature map and channels.

    Parameters
    ----------
    in_features: int

        Number of input features.

    out_features: int

        Number of output features.
    """

    def __init__(self, in_features, out_features):
        super(_Up, self).__init__()

        self.deconv3d = nn.ConvTranspose3d(in_features, out_features, 4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm3d(out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        out = self.deconv3d(x)
        out = self.in1(out)
        out = self.relu(out)

        return out

class _UpConvolution(nn.Module):
    """
    'Up-Convolution' Block

    Basic3d x n + 'Up'

    Parameters
    ----------
    in_features: int

        Number of input features.

    out_features: int

        Number of output features.

    n: int 

        Number of Basic3d blocks. 
    """
    def __init__(self, in_features, out_features, n):
        super(_UpConvolution, self).__init__()

        self.features = nn.Sequential(OrderedDict([]))
        block = _Basic3d(in_features, out_features, n)

        self.features.add_module('basic3d%d' % n, block)
        self.up_module = _Up(out_features, out_features)
    
    def forward(self, x):

        out = self.features(x)
        out = self.up_module(out)

        return out

class Encoder(nn.Module):
    """
    Encoder Module. 

    Returns a list with 6 outputs. 1 for ConnectionA and 5 for ConnectionB.

    Parameters
    ----------
    num_init_features: int

        Initial number of features. 
    
    growth_rate: int, default = 32
    """
    def __init__(self, num_init_features, growth_rate=32): #, num_dense=5
        super(Encoder, self).__init__()

        num_layers = 6 # number of dense layers in 'Dense' Module.
        self.num_dense = 5 # number of 'Dense' Modules. 
        self.basic2d = _Basic2d(1, num_init_features)
        num_features = num_init_features
        self.encoder_blocks = nn.ModuleList([])
        for i in range(self.num_dense):
            block = _Dense(num_features, 1, growth_rate=growth_rate)
            self.encoder_blocks.append(block)
            num_features = int((num_features + num_layers * growth_rate) * 0.5)

    def forward(self, x):

        b_list = []
        out = self.basic2d(x)
        for i in range(self.num_dense):
            b_list.append(out)
            out = self.encoder_blocks[i](out)
        out = F.avg_pool2d(out, 2)

        return out, b_list


class Decoder(nn.Module):
    """
    Decoder Module. 

    Takes output list of Encoder Module and returns a list with 5 outputs tensors.  
    
    Parameters: 
    -----------
    num_init_features: int

        Number of features coming out of ConnectionA. 

    feature_list: list(int)

        List of features of all tensors coming out of ConnectionB. 

    n: int
    
        Number of Basic3d layers (xN in paper).

    out_features: int

        Number of output features after last Basic3d
    """
    def __init__(self, num_init_features, feature_list, n, out_features=32):
        super(Decoder, self).__init__()

        self.up_module = _Up(num_init_features, feature_list[0])
        self.decoder_basic3d_xN = nn.ModuleList([])
        #Up_Conv block
        for i in range(4):
            block = _Basic3d(feature_list[i]*2, feature_list[i], n)
            self.decoder_basic3d_xN.append(block)

        self.decoder_up = nn.ModuleList([])
        for i in range(4):
            block = _Up(feature_list[i], feature_list[i+1])
            self.decoder_up.append(block)
        
        self.basic3d = _Basic3d(feature_list[4]*2, out_features)

    def forward(self, x, b_list):

        decoder_output_list = []
        decoder_output_list.append(x)
        out = self.up_module(x)
        print(out.shape)
        print(b_list[0].shape)
        print(b_list[1].shape)
        print(b_list[2].shape)
        print(b_list[3].shape)
        print(b_list[4].shape)
        for i in range(5):
            out = torch.cat([out, b_list[i]], 1)
            if i < 4:
                out = self.decoder_basic3d_xN[i](out)
                decoder_output_list.append(out)
                out = self.decoder_up[i](out)
            else: 
                out = self.basic3d(out)
                decoder_output_list.append(out)

        return decoder_output_list

class Encoder_Decoder(nn.Module):
    """
    Encoder-Decoder Module. 

    Connects Encoder and Decoder. Returns List of 6 output tensors for Fusion Module. 

    a_magic_number = length of flattend tensor for connection A.

    Parameters
    ----------
    batch_size : int
        
        Batch size of dataset. 
    
    growth_rate : int, default = 32
    """
    def __init__(self, batch_size, growth_rate=32):
        super(Encoder_Decoder, self).__init__()
        # growth_rate = 16
        bs = batch_size
        output_dim = 4
        layers = 6
        b_channel_list = [180, 168, 144, 96, 1]
        # b_channel_list = b_channel_list * bs

        # encoder_output_channels = 186 * bs
        a_magic_number = 744 * bs
        decoder_input_channels = a_magic_number//(output_dim**3)
        layers_b = 1
        self.encoder = Encoder(1, growth_rate=growth_rate)

        self.connection_b_modules = nn.ModuleList()
        
        for i in reversed(range(len(b_channel_list))):
            module = ConnectionB(b_channel_list[i],                 b_channel_list[i])
            self.connection_b_modules.append(module)
        
        output_size = output_dim**3 * bs
        # output_size = a_magic_number - a_magic_number%int(output_dim**3)
        c = output_size//(bs*output_dim**3)
        output_shape = [bs,  1, output_dim, output_dim, output_dim]

        self.connection_a = ConnectionA(a_magic_number, output_size, output_shape)
        self.decoder = Decoder(1, b_channel_list, layers_b)
        
    def forward(self, input_a):
        """
        Parameters
        ----------
        input_a : tensor

            Tensor of input image with size 128x128.
        """
        encoder_output, connection_b_input = self.encoder(input_a)
        b_output = []
        for i in reversed(range(len(connection_b_input))):
            temp = self.connection_b_modules[i](connection_b_input[i], connection_b_input[i].shape[3])
            b_output.append(temp)

        encoder_output = encoder_output.view(-1)   

        a_output = self.connection_a(encoder_output)
        decoder_result = self.decoder(a_output, b_output)
        return decoder_result

class Fusion(nn.Module):
    """
    Fusion Module. 
    
    Takes input from two auto encoders with a list of 6 tensors each. Middle network in x2ct paper Fig3.

    Parameters
    ---------- 
    batch_size : int

        Batch size of dataset
    """
    def __init__(self, batch_size):
        
        super(Fusion, self).__init__()

        self.con_c = ConnectionC()
        num_init_features = 1
        feature_list = [180, 168, 144, 96, 32]
        # feature_list = feature_list * batch_size
        n = 1

        # out_features = 32
        self.up_module = _Up(num_init_features, feature_list[0])
        self.up_conv_list = []
        self.fusion_blocks = nn.ModuleList([])

        for i in range(4):
            block = _UpConvolution(feature_list[i]*2, feature_list[i+1], n)
            self.fusion_blocks.append(block)

        self.basic3d = _Basic3d(feature_list[4]*2, 1) 

    def forward(self, in_x, in_y):
        """
        Parameters
        ----------
        in_x : tensor

            Connection A output of Decoder A 

        in_y: tensor

            Connection A output of Decoder B
        """
        out = self.con_c(in_x[0], in_y[0])
        # pop first tensor from lists after use
        x = in_x[1:]
        y = in_y[1:]
        out = self.up_module(out)
        # ConnectionC for inputs from encoder A and B and concat with last out. Then run through fusion_blocks and at the end through basic3d.
        for i in range(5): 

            temp = self.con_c(x[i], y[i])
            out = torch.cat([out, temp], 1)
            if i < 4: 
                out = self.fusion_blocks[i](out)
            else: 
                out = self.basic3d(out)

        return out

class CT_Discriminator(nn.Module):
    """
    Discriminator for CT arrays. 

    Returns 1 or 0.

    Parameters
    ----------
    batch_size: int

        Batch size of dataset. 
    """
    def __init__(self, batch_size):
        super(CT_Discriminator, self).__init__()

        self.patch3d = nn.Sequential(OrderedDict([]))
        for i in range(3):
            if i == 0:
                block1 = nn.Conv3d(batch_size,64, 4, stride=2)
            else: block1 = nn.Conv3d(64,64, 4, stride=2)
            block2 = nn.InstanceNorm3d(64)
            block3 = nn.ReLU()
            self.patch3d.add_module('conv3d%d' % (i+1), block1)
            self.patch3d.add_module('in%d' % (i+1), block2)
            self.patch3d.add_module('relu%d' % (i+1), block3)
        self.patch3d.add_module('conv3d4', nn.Conv3d(64, 32, 4, stride=1))
        self.patch3d.add_module('in4', nn.InstanceNorm3d(32))
        self.patch3d.add_module('relu4', nn.ReLU())
        self.patch3d.add_module('conv3d5', nn.Conv3d(32, 1, 4))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # returns [1,1,8,8,8] array
        out = self.patch3d(x)
        out = self.sig(out)
        return out

class X_Discriminator(nn.Module):
    def __init__(self, batch_size):
        super(X_Discriminator, self).__init__()

        self.patch2d = nn.Sequential(OrderedDict([]))
        for i in range(3):
            if i == 0:
                block1 = nn.Conv2d(batch_size,64, 4, stride=2)
            else: block1 = nn.Conv2d(64,64, 4, stride=2)
            block2 = nn.InstanceNorm3d(64)
            block3 = nn.ReLU()
            self.patch3d.add_module('conv2d%d' % (i+1), block1)
            self.patch3d.add_module('in%d' % (i+1), block2)
            self.patch3d.add_module('relu%d' % (i+1), block3)
        self.patch3d.add_module('conv2d4', nn.Conv3d(64, 32, 4, stride=1))
        self.patch3d.add_module('in4', nn.InstanceNorm3d(32))
        self.patch3d.add_module('relu4', nn.ReLU())
        self.patch3d.add_module('conv2d5', nn.Conv3d(32, 1, 4))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # returns [1,1,8,8,8] array
        out = self.patch2d(x)
        out = self.sig(out)
        return out
        
class Generator(nn.Module):
    def __init__(self, batch_size):
        super(Generator, self).__init__()
        self.gen_a = Encoder_Decoder(batch_size)
        self.gen_b = Encoder_Decoder(batch_size)
        self.gen_fusion = Fusion(batch_size)

    def forward(self, x1, x2):
        x1 = x1.float()
        x2 = x2.float()
        x1 = self.gen_a(x1)
        x2 = self.gen_b(x2)
        output = self.gen_fusion(x1, x2)

        return output

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
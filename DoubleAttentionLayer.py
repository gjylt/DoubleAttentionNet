from torch import nn
import torch
from torch.autograd import Variable


class DoubleAttentionLayer(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1 ):
        super(DoubleAttentionLayer, self).__init__()

        self.K           = k
        self.c_m = c_m
        self.c_n = c_n
        self.softmax     = nn.Softmax()
        self.in_channels = in_channels

        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)

    def forward(self, x):

        b, c, h, w = x.size()

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        tmpA = A.view( batch, self.K, self.c_m, h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*h*w )
        tmpB = B.view( batch, self.K, self.c_n, h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*h*w )
        tmpV = V.view( batch, self.K, self.c_n, h*w ).permute(0,1,3,2).contiguous().view( int(b*h*w), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB )      #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV ) #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view(batch, self.c_m, self.K,h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, h, w )

        return tmpZ


if __name__ == "__main__":


    # tmp1        = torch.ones(2,2,3)
    # tmp1[1,:,:] = tmp1[1,:,:]*2
    # tmp2 = tmp1.permute(0,2,1)
    # print(tmp1)
    # print( tmp2)
    # print( tmp1.matmul(tmp2))

    in_channels = 10
    c_m = 4
    c_n = 3

    doubleA = DoubleAttentionLayer(in_channels, c_m, c_n)

    x   = torch.ones(2,in_channels,6,8)
    x   = Variable(x)
    tmp = doubleA(x)

    print("result")

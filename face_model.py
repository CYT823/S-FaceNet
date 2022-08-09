from torch.nn import Linear, Conv2d, BatchNorm2d, ReLU, Module, Parameter
from torch import nn
import torch
import math

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

##################################  MobileFaceNet #############################################################
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.relu = ReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        self.num_block = num_block
        self.residual = Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups)

        # modules = []
        # for _ in range(num_block):
        #     modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        # self.model = Sequential(*modules)
    def forward(self, x):
        for _ in range(self.num_block):
            x = self.residual(x)
        return x
        # return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        '''
        # Original Architecture
        self.conv1      = Conv_block(in_c=3, out_c=64, groups=1, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw   = Conv_block(in_c=64, out_c=64, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23    = Depth_Wise(in_c=64, out_c=64, groups=128, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_3     = Residual(c=64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34    = Depth_Wise(in_c=64, out_c=128, groups=256, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_4     = Residual(c=128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45    = Depth_Wise(in_c=128, out_c=128, groups=512, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_5     = Residual(c=128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(in_c=128, out_c=512, groups=1, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw  = Linear_block(in_c=512, out_c=512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        '''

        '''
        # Smallest Architecture - feature vector 128 
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(4, 4), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(4, 4), padding=(1, 1), groups=128)
        self.conv3_rd = Residual(c=128, num_block=3, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_st = Conv_block(128, 256, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv5_li = Linear_block(256, 128, groups=128, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv6_flatten = Flatten()
        self.linear = Linear(128, embedding_size, bias=False)
        '''

        '''
        # Smallest Architecture - feature vector 256
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(4, 4), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(4, 4), padding=(1, 1), groups=128)
        self.conv3_rd = Residual(c=128, num_block=3, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_st = Conv_block(128, 256, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv5_li = Linear_block(256, 256, groups=256, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv6_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''
        
        '''
        # Smallest Architecture - feature vector 512
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(4, 4), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(4, 4), padding=(1, 1), groups=128)
        self.conv3_rd = Residual(c=128, num_block=3, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_st = Conv_block(128, 256, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv5_li = Linear_block(256, 512, groups=256, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        '''

        '''
        # Bigger Architecture 1 - feature vector 256 _ DW + DW + Res
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv3_dw = Depth_Wise(in_c=128, out_c=256, residual=False, kernel=(3, 3), stride=(3, 3), padding=(1, 1), groups=128)
        self.conv4_rd = Residual(c=256, num_block=2, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_st = Conv_block(256, 512, kernel=(1, 1), stride=(4, 4), padding=(0, 0))
        self.conv6_li = Linear_block(512, 256, groups=256, kernel=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv7_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''

        '''
        # Bigger Architecture 2 - feature vector 256 _ DW + Res + Li + Res
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(3, 3), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv3_rd = Residual(c=128, num_block=2, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_li = Linear_block(128, 256, groups=128, kernel=(3, 3), stride=(2, 2), padding=(0, 0))
        self.conv5_rd = Residual(c=256, num_block=2, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6_st = Conv_block(256, 512, kernel=(1, 1), stride=(4, 4), padding=(0, 0))
        self.conv7_li = Linear_block(512, 256, groups=256, kernel=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv8_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''

        '''
        # Bigger Architecture 3 - feature vector 256 _ 2*(DW + Res)
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(3, 3), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=64, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv3_rd = Residual(c=64, num_block=2, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv5_rd = Residual(c=128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6_st = Conv_block(in_c=128, out_c=256, kernel=(1, 1), stride=(4, 4), padding=(0, 0))
        self.conv7_li = Linear_block(in_c=256, out_c=256, groups=256, kernel=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv8_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''

        '''
        # Bigger Architecture 4 - feature vector 256 _ 3*(DW + Res)
        self.conv1_st = Conv_block(in_c=3, out_c=64, kernel=(3, 3), stride=(3, 3), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=64, out_c=64, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv3_rd = Residual(c=64, num_block=2, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_dw = Depth_Wise(in_c=64, out_c=128, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv5_rd = Residual(c=128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6_dw = Depth_Wise(in_c=128, out_c=256, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv7_rd = Residual(c=256, num_block=2, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8_st = Conv_block(in_c=256, out_c=256, kernel=(1, 1), stride=(2, 2), padding=(0, 0))
        self.conv9_li = Linear_block(in_c=256, out_c=256, groups=256, kernel=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv10_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''

        '''
        # Smallest Architecture 2 - feature vector 256
        self.conv1_st = Conv_block(in_c=3, out_c=32, kernel=(5, 5), stride=(5, 5), padding=(1, 1))
        self.conv2_dw = Depth_Wise(in_c=32, out_c=64, groups=128, residual=False, kernel=(3, 3), stride=(3, 3), padding=(1, 1))
        self.conv3_rd = Residual(c=64, num_block=3, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_st = Conv_block(in_c=64, out_c=128, kernel=(1, 1), stride=(3, 3), padding=(0, 0))
        self.conv5_li = Linear_block(in_c=128, out_c=256, groups=128, kernel=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv6_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''

        '''
        # Smallest Architecture 3 - feature vector 256
        # need more channel if the kernel and stride too big?
        # try to give more layer to fix this problem
        self.conv1_st = Conv_block(in_c=3, out_c=32, groups=1, kernel=(3, 3), stride=(2, 2), padding=(0, 0))
        self.conv2_li = Linear_block(in_c=32, out_c=64, groups=32, kernel=(3, 3), stride=(3, 3), padding=(0, 0))
        self.conv3_dw = Depth_Wise(in_c=64, out_c=64, groups=128, kernel=(3, 3), stride=(3, 3), padding=(1, 1), residual=False)
        self.conv4_rd = Residual(c=64, num_block=3, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_st = Conv_block(in_c=64, out_c=128, groups=1, kernel=(1, 1), stride=(2, 2), padding=(0, 0))
        self.conv6_li = Linear_block(in_c=128, out_c=256, groups=128, kernel=(3, 3), stride=(2, 2), padding=(0, 0))
        self.conv7_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        '''

        ''''''
        # Smallest Architecture 4 - feature vector 256
        # using group convolution to replace standard one and to stable the state
        self.conv1_st = Conv_block(in_c=3, out_c=64, groups=1, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_gr = Conv_block(in_c=64, out_c=64, groups=64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))      # Group Conv.
        self.conv3_dw = Depth_Wise(in_c=64, out_c=128, groups=128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), residual=False)
        self.conv4_gr = Conv_block(in_c=128, out_c=128, groups=128, kernel=(3, 3), stride=(2, 2), padding=(1, 1))   # Group Conv.
        self.conv5_rd = Residual(c=128, num_block=2, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6_pw = Conv_block(in_c=128, out_c=256, groups=1, kernel=(1, 1), stride=(1, 1), padding=(0, 0))     # PointWise
        self.conv7_gr = Conv_block(in_c=256, out_c=256, groups=256, kernel=(3, 3), stride=(2, 2), padding=(0, 0))
        self.conv8_li = Linear_block(in_c=256, out_c=256, groups=256, kernel=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv9_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        

        '''
        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
        
    def forward(self, x):
        '''
        # Original Architecture Flow
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        '''

        '''
        # Smallest Architecture Flow 1 & 2
        out = self.conv1_st(x)
        out = self.conv2_dw(out)
        out = self.conv3_rd(out)
        out = self.conv4_st(out)
        out = self.conv5_li(out)
        out = self.conv6_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        '''

        '''
        # Smallest Architecture Flow 3
        out = self.conv1_st(x)
        out = self.conv2_li(out)
        out = self.conv3_dw(out)
        out = self.conv4_rd(out)
        out = self.conv5_st(out)
        out = self.conv6_li(out)
        out = self.conv7_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        '''

        ''''''
        # Smallest Architecture 4
        out = self.conv1_st(x)
        out = self.conv2_gr(out)
        out = self.conv3_dw(out)
        out = self.conv4_gr(out)
        out = self.conv5_rd(out)
        out = self.conv6_pw(out)
        out = self.conv7_gr(out)
        out = self.conv8_li(out)
        out = self.conv9_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        

        '''
        # Bigger Architecture Flow 1
        out = self.conv1_st(x)
        out = self.conv2_dw(out)
        out = self.conv3_dw(out)
        out = self.conv4_rd(out)
        out = self.conv5_st(out)
        out = self.conv6_li(out)
        out = self.conv7_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        '''

        '''
        # Bigger Architecture Flow 2
        out = self.conv1_st(x)
        out = self.conv2_dw(out)
        out = self.conv3_rd(out)
        out = self.conv4_li(out)
        out = self.conv5_rd(out)
        out = self.conv6_st(out)
        out = self.conv7_li(out)
        out = self.conv8_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        '''

        '''
        # Bigger Architecture Flow 3
        out = self.conv1_st(x)
        out = self.conv2_dw(out)
        out = self.conv3_rd(out)
        out = self.conv4_dw(out)
        out = self.conv5_rd(out)
        out = self.conv6_st(out)
        out = self.conv7_li(out)
        out = self.conv8_flatten(out)
        out = self.linear(out)
        '''

        '''
        # Bigger Architecture Flow 4
        out = self.conv1_st(x)
        out = self.conv2_dw(out)
        out = self.conv3_rd(out)
        out = self.conv4_dw(out)
        out = self.conv5_rd(out)
        out = self.conv6_dw(out)
        out = self.conv7_rd(out)
        out = self.conv8_st(out)
        out = self.conv9_li(out)
        out = self.conv10_flatten(out)
        out = self.linear(out)
        out = l2_norm(out)
        '''

        return out

##################################  Arcface head #############################################################
class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        nn.init.xavier_uniform_(self.kernel)
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0) # normalize for each column
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
        # output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

if __name__ == "__main__":
    import time
    
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.rand(32, 3, 112, 112).to(device)

    print("Loading MobileFaceNet...")
    net = MobileFaceNet(embedding_size=256).to(device)
    
    print("Loading ArcFace Model...")
    margin = Arcface(embedding_size=256, classnum=10, s=32., m=0.5).to(device)

    # x = net(input)
    # output = margin(x, 7)

    end = time.time()
    print("Total time: {}".format(end-start))

    # Show summary of the model
    from torchinfo import summary
    summary(net, input_size=(1, 3, 112, 112))

    '''
    #Testing Pruning Start

    module = net.conv1.conv

    # print(list(module.named_modules()),"\n")
    # print(list(module.named_buffers()),"\n")
    # print(list(module.named_parameters()),"\n")

    # 前
    # print(module.weight, "\n")
    f=open('a1.txt','w')
    for ele in list(module.weight):
        f.write(str(ele) + '\n')
    f.close()
    
    # 後
    import torch.nn.utils.prune as prune
    prune.random_unstructured(module, name='weight', amount=0.3)
    
    # print(module.weight, "\n")
    f=open('a2.txt','w')
    for ele in list(module.weight):
        f.write(str(ele) + '\n')
    f.close()

    # Testing Pruning End
    '''


    ''' There are three ways to plot the architecture of model
      1. TorchViz       - create vertical view of architecture
      2. Hiddenlayer    - create landscape view of architecture
      3. Netron         - draw simplest way & need to install an app on your devices
    '''
    # from torchviz import make_dot
    # dot = make_dot(x.mean(), params=dict(net.named_parameters()))
    # dot.format = "png"
    # dot.render("MobileFaceNet Architecture")

    # import hiddenlayer as hl
    # transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
    # graph = hl.build_graph(net, torch.rand(32, 3, 112, 112), transforms=transforms)
    # graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save('rnn_hiddenlayer', format='png')

    # Saving model which is used for Netron
    torch.save(net,'weight.pth')
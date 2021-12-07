import torch
import torch.nn as nn
from torchvision import models 
from torch.nn import init
from torchsummary import summary 


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_dim, output_ch=1):
        super(U_Net,self).__init__()
        f = 32
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_dim,ch_out=f)
        self.Conv2 = conv_block(ch_in=f,ch_out=f*2)
        self.Conv3 = conv_block(ch_in=f*2,ch_out=f*4)
        self.Conv4 = conv_block(ch_in=f*4,ch_out=f*8)
        self.Conv5 = conv_block(ch_in=f*8,ch_out=f*16)

        self.Up5 = up_conv(ch_in=f*16,ch_out=f*8)
        self.Up_conv5 = conv_block(ch_in=f*16, ch_out=f*8)

        self.Up4 = up_conv(ch_in=f*8,ch_out=f*4)
        self.Up_conv4 = conv_block(ch_in=f*8, ch_out=f*4)
        
        self.Up3 = up_conv(ch_in=f*4,ch_out=f*2)
        self.Up_conv3 = conv_block(ch_in=f*4, ch_out=f*2)
        
        self.Up2 = up_conv(ch_in=f*2,ch_out=f)
        self.Up_conv2 = conv_block(ch_in=f*2, ch_out=f)

        self.Conv_1x1 = nn.Conv2d(f,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_dim,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_dim,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self,img_dim, out_dim=256, features=32): #org 64
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_dim,ch_out=features)
        self.Conv2 = conv_block(ch_in=features,ch_out=features*2)
        self.Conv3 = conv_block(ch_in=features*2, ch_out=features*4)
        self.Conv4 = conv_block(ch_in=features*4,ch_out=features*8)
        self.Conv5 = conv_block(ch_in=features*8,ch_out=features*16)
        
        self.avrgpool = nn.AdaptiveAvgPool2d((2, 2))  
        self.fc = nn.Linear(features*16*4, out_dim) 

        self.Up5 = up_conv(ch_in=features*16,ch_out=features*8)
        self.Att5 = Attention_block(F_g=features*8,F_l=features*8,F_int=features*4)
        self.Up_conv5 = conv_block(ch_in=features*16, ch_out=features*8)

        self.Up4 = up_conv(ch_in=features*8,ch_out=features*4)
        self.Att4 = Attention_block(F_g=features*4,F_l=features*4,F_int=features*2)
        self.Up_conv4 = conv_block(ch_in=features*8, ch_out=features*4)
        
        self.Up3 = up_conv(ch_in=features*4,ch_out=features*2)
        self.Att3 = Attention_block(F_g=features*2,F_l=features*2,F_int=features)
        self.Up_conv3 = conv_block(ch_in=features*4, ch_out=features*2)
        
        self.Up2 = up_conv(ch_in=features*2,ch_out=features)
        self.Att2 = Attention_block(F_g=features,F_l=features,F_int=features//2)
        self.Up_conv2 = conv_block(ch_in=features*2, ch_out=features)

        self.Conv_1x1 = nn.Conv2d(features,img_dim,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        embd = self.avrgpool(x5)
        embd = embd.view(embd.size(0), -1)
        embd = self.fc(embd)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1 #,embd 


class R2AttU_Net(nn.Module):
    def __init__(self,img_dim, out_dim = 256, features = 32, t=2): #origin 64
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_dim,ch_out=features,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=features,ch_out=features*2,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=features*2,ch_out=features*4,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=features*4,ch_out=features*8,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=features*8,ch_out=features*16,t=t)
        
        self.avrgpool = nn.AdaptiveAvgPool2d((2, 2))  
        self.fc = nn.Linear(features*8*4, out_dim) 

        self.Up5 = up_conv(ch_in=features*16,ch_out=features*8)
        self.Att5 = Attention_block(F_g=features*8,F_l=features*8,F_int=features*4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=features*16, ch_out=features*8,t=t)
        
        self.Up4 = up_conv(ch_in=features*8,ch_out=features*4)
        self.Att4 = Attention_block(F_g=features*4,F_l=features*4,F_int=features*2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=features*8, ch_out=features*4,t=t)
        
        self.Up3 = up_conv(ch_in=features*4,ch_out=features*2)
        self.Att3 = Attention_block(F_g=features*2,F_l=features*2,F_int=features)
        self.Up_RRCNN3 = RRCNN_block(ch_in=features*4, ch_out=features*2,t=t)
        
        self.Up2 = up_conv(ch_in=features*2,ch_out=features)
        self.Att2 = Attention_block(F_g=features,F_l=features,F_int=features//2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=features*2, ch_out=features,t=t)

        self.Conv_1x1 = nn.Conv2d(features,img_dim,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
 
        #x5 = self.Maxpool(x4)
        #x5 = self.RRCNN5(x5)

        embd = self.avrgpool(x4)
        embd = embd.view(embd.size(0), -1)
        embd = self.fc(embd)

        # decoding + concat path
        #d5 = self.Up5(x5)
        #x4 = self.Att5(g=d5,x=x4)
        #d5 = torch.cat((x4,d5),dim=1)
        #d5 = self.Up_RRCNN5(d5)
 
        d4 = self.Up4(x4) 
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, embd


class Mapper(nn.Module):
    def __init__(self, pre_network, join_type, img_dim, out_dim=256):
        super().__init__()

        self.join_type = join_type

        # let's start with encoder
        model = getattr(models, pre_network)(pretrained=False)
        model.conv1 = nn.Conv2d(img_dim, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        
        if self.join_type == "concat":
            model.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 2))

        model = list(model.children())[:-1]
        self.backbone = nn.Sequential(*model)

        if pre_network == "resnet50": nfc = 2048
        elif pre_network == "resnet18": nfc = 512 

        if self.join_type == "concat":
            self.fc1 = nn.Linear(nfc*2, out_dim)
        else:
            self.fc1 = nn.Linear(nfc, out_dim)

        # now decoder
        decoder = [] 
        in_channels = nfc 
        out_channels = in_channels // 2

        num_blocks = 8 
        for _ in range(num_blocks):
            decoder += [
                nn.ConvTranspose2d(in_channels, out_channels, 
                        kernel_size=3, stride=2, 
                        padding=1, output_padding=1),

                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]

            in_channels = out_channels
            out_channels = in_channels // 2

 
        decoder += [nn.ReflectionPad2d(3), 
                    nn.Conv2d(out_channels*2, img_dim, kernel_size=7),
                    nn.Tanh()]
    
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x):
        x = self.backbone(x)
        img =  self.decoder(x)

        x = x.view(x.size(0), -1)
        embedding = self.fc1(x)

        return img, embedding

    def EncodeImage(self, x):
        return self.backbone(x)

    def DecodeImage(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, join_type, in_channels, img_size=256):
        super().__init__()

        self.join_type = join_type 
        self.shared_conv = nn.Sequential(
            *self.discriminator_block(in_channels, 16, bn=False),
            *self.discriminator_block(16, 32),
            *self.discriminator_block(32, 64),
            *self.discriminator_block(64, 128)
        )

        #image pixel in down-sampled features map
        if self.join_type == "concat":
            input_node = 128 * (img_size // (2**4)) * (img_size // (2**3))
        else:
            input_node = 128 * (img_size // (2**4))**2 

        self.D1 = nn.Linear(input_node, 1)

    def discriminator_block(self, in_filters, out_filters, bn=True):

        block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]

        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        block.extend([nn.LeakyReLU(0.2, inplace=True), 
                        nn.Dropout2d(0.25)])

        return block

    def forward(self, x):
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)
        x = self.D1(x)
        return x 



if __name__ == "__main__":
    model = U_Net(img_dim=1).cuda() 
    
    #print(model.state_dict()["Conv1.conv.0.bias"].size())
    #print(model.Conv_1x1)
    #print(model.state_dict()["Conv1.conv.0.weight"].size())
    summary(model, (1, 256, 256), device="cpu") 

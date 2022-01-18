import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        ngf = 28
        nc = 3
        nz = 100
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz+10, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf*32, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf*16, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf*8, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf*4, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

        )
        self.out=nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.main(x)
        x = self.out(x)
        return x 

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        nc = 3
        ndf = 28
        self.ndf = ndf
        self.layer1=nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
            
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf, 3, 2, 1, bias=False),
        )



        self.linear_dis = nn.Linear(ndf, 1) # True or Fake
        self.linear_aux = nn.Linear(ndf, 10) # which digit
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, self.ndf)
        # print("here", x.shape) # [128, 28]
        dis = self.linear_dis(x) 
        aux = self.linear_aux(x)
        real_or_fake = self.sigmoid(dis)
        digit = self.softmax(aux)
        return real_or_fake, digit
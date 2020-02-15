# EVA-4-Session-4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)      # Input 1, No. of Kernels 8, Kernel Size 3*3, Bias = False
        self.BN1 = nn.BatchNorm2d(8)                                #Performing BatchNormalization of 8 Output Channels
        nn.Dropout2d(0.25)                                          #Dropout(0.25)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)     #Input 8, No. of Kernels 16, Kernel Size 3*3, Bias = False
        self.BN2 =nn.BatchNorm2d(16)                                #Performing BatchNormalization of 16 Output Channels
        nn.Dropout2d(0.25)                                          #Dropout(0.25)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 16, 3, padding=1,bias=False)     #Input 16, No. of Kernels 16, Kernel Size 3*3, Bias = False
        self.BN3 =nn.BatchNorm2d(16)                                #Performing BatchNormalization of 16 Output Channels
        nn.Dropout2d(0.25)                                          #Dropout(0.25)    

        self.conv4 = nn.Conv2d(16, 24, 3, padding=1,bias=False)     #Input 16, No. of Kernels 24, Kernel Size 3*3, Bias = False
        self.BN4 =nn.BatchNorm2d(24)                                #Performing BatchNormalization of 24 Output Channels
        self.D2 = nn.Dropout2d(0.25)                                #Added Dropout(0.25) after 4 Convolutions

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(24, 32, 3, padding=1,bias=False)     #Input 24, No. of Kernels 32, Kernel Size 3*3, Bias = False
                          
        self.BN5 =nn.BatchNorm2d(32)                                #Performing BatchNormalization of 32 Output Channels
        nn.Dropout2d(0.25)                                          #Dropout(0.25)

        self.conv6 = nn.Conv2d(32, 10, 1,bias=False)                #Input 32, No. of Kernels 10, Kernel Size 1*1, Bias = False (Flatten)
        self.conv7 = nn.Conv2d(10, 10, 7,bias=False)                #Input 10, No. of Kernels 10, GAP of 7 convolution layers 7, Bias = False

    def forward(self, x):
        x = self.pool1(F.relu(self.BN2(self.conv2(F.relu(self.BN1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.D2(self.BN4(self.conv4(F.relu(self.BN3(self.conv3(x))))))))
        x = F.relu(self.conv6(F.relu(self.BN5(self.conv5(x)))))
        x = (self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
        




Requirement already satisfied: torchsummary in /usr/local/lib/python3.6/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
       BatchNorm2d-2            [-1, 8, 28, 28]              16
            Conv2d-3           [-1, 16, 28, 28]           1,152
       BatchNorm2d-4           [-1, 16, 28, 28]              32
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 16, 14, 14]           2,304
       BatchNorm2d-7           [-1, 16, 14, 14]              32
            Conv2d-8           [-1, 24, 14, 14]           3,456
       BatchNorm2d-9           [-1, 24, 14, 14]              48
        Dropout2d-10           [-1, 24, 14, 14]               0
        MaxPool2d-11             [-1, 24, 7, 7]               0
           Conv2d-12             [-1, 32, 7, 7]           6,912
      BatchNorm2d-13             [-1, 32, 7, 7]              64
           Conv2d-14             [-1, 10, 7, 7]             320
           Conv2d-15             [-1, 10, 1, 1]           4,900
================================================================
Total params: 19,308
Trainable params: 19,308
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.50
Params size (MB): 0.07
Estimated Total Size (MB): 0.58
---------------------------------------------------------------



model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 20):
    print("****************************************epoch{}".format(epoch))
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

import torch

class ResNet_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet_Block, self).__init__()

        # Note: conv1d expects inputs of shape (batch_size, channels, length)
        # So in our case (batch_size, 1, len(datapoint))
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=out_channels)
        self.relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(num_features=out_channels)

        self.conv3 = torch.nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(num_features=in_channels)

        self.conv4 = torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1)
        self.bn4 = torch.nn.BatchNorm1d(num_features=in_channels)

        # Downsample if an odd stride shape causes mismatch in shapes that would prevent skip connection addition
        self.downsample = None
        if stride != 1 or in_channels != out_channels: 
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm1d(in_channels)
            )
    
    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample:
            identity = self.downsample(identity)
        
        x += identity

        x = self.relu(x)

        return x

# Deep Neural Network - ResNet block, BiGRU block, attention layer, inference layer in the paper
# We'll start with just ResNet depending on time constraints
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.num_classes = 2
        self.input_channels = 1
        self.output_channels = 4

        # ResNet
        self.conv = torch.nn.Conv1d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm1d(num_features=self.input_channels)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.res_block1 = ResNet_Block(self.input_channels, self.output_channels, stride=1)
        self.res_block2 = ResNet_Block(self.output_channels, self.output_channels, stride=1)
        self.res_block3 = ResNet_Block(self.output_channels, self.output_channels, stride=1)

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(self.output_channels, self.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.avgpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x       
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, InceptionBlock


class GenerativeInceptionNN(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeInceptionNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.drop1 = nn.Dropout(p=0.3)

        self.inception1 = InceptionBlock(in_channels=channel_size * 4, num_1x1=36, num_3x3=68
                                         , num_5x5=16, num_pool=8, reduce3=32, reduce5=8)
        self.bni1 = nn.BatchNorm2d(channel_size*4)

        self.inception2 = InceptionBlock(in_channels=channel_size * 4, num_1x1=36, num_3x3=68
                                         , num_5x5=16, num_pool=8, reduce3=32, reduce5=8)
        self.bni2 = nn.BatchNorm2d(channel_size * 4)

        self.inception3 = InceptionBlock(in_channels=channel_size * 4, num_1x1=36, num_3x3=68
                                         , num_5x5=16, num_pool=8, reduce3=32, reduce5=8)
        self.bni3 = nn.BatchNorm2d(channel_size * 4)

        self.drop2=nn.Dropout(p=0.3)

        self.inception4 = InceptionBlock(in_channels=channel_size * 4, num_1x1=36, num_3x3=68
                                         , num_5x5=16, num_pool=8, reduce3=32, reduce5=8)
        self.bni4 = nn.BatchNorm2d(channel_size * 4)

        self.inception5 = InceptionBlock(in_channels=channel_size * 4, num_1x1=36, num_3x3=68
                                         , num_5x5=16, num_pool=8, reduce3=32, reduce5=8)
        self.bni5 = nn.BatchNorm2d(channel_size * 4)

        self.drop3 = nn.Dropout(p=0.5)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size*2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size*2)

        self.conv5 = nn.ConvTranspose2d(channel_size*2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.drop1(x)

        x = self.bni1(self.inception1(x))
        x = self.bni2(self.inception2(x))
        x = self.bni3(self.inception3(x))

        x = self.drop2(x)

        x = self.bni4(self.inception4(x))
        x = self.bni5(self.inception5(x))

        x = self.drop3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = F.sigmoid(self.pos_output(self.dropout_pos(x)))
            cos_output = F.tanh(self.cos_output(self.dropout_cos(x)))
            sin_output = F.tanh(self.sin_output(self.dropout_sin(x)))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = F.sigmoid(self.pos_output(x))
            cos_output = F.tanh(self.cos_output(x))
            sin_output = F.tanh(self.sin_output(x))
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

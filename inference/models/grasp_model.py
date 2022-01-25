import torch.nn as nn
import torch.nn.functional as F
import torch


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)


        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, num_1x1, num_3x3, num_5x5, num_pool, reduce3, reduce5):
        super(InceptionBlock, self).__init__()
        self.conv_cat1 = nn.Conv2d(in_channels, out_channels=num_1x1, kernel_size=1)
        self.conv_cat2pre = nn.Conv2d(in_channels, out_channels=reduce3, kernel_size=1)
        self.conv_cat2 = nn.Conv2d(in_channels=reduce3, out_channels=num_3x3, kernel_size=3, padding=1)
        self.conv_cat3pre = nn.Conv2d(in_channels, out_channels=reduce5, kernel_size=1)
        self.conv_cat3 = nn.Conv2d(in_channels=reduce5, out_channels=num_5x5, kernel_size=5, padding=2)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_cat4 = nn.Conv2d(in_channels=in_channels, out_channels=num_pool, kernel_size=1)

    def forward(self, x_in):
        x1 = F.relu(self.conv_cat1(x_in))
        x2 = F.relu(self.conv_cat2pre(x_in))
        x2 = F.relu(self.conv_cat2(x2))
        x3 = F.relu(self.conv_cat3pre(x_in))
        x3 = F.relu(self.conv_cat3(x3))
        x4 = self.mp(x_in)
        x4 = F.relu(self.conv_cat4(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return F.relu(x + x_in)


class InceptionBlock2a(nn.Module):

    def __init__(self, in_channels, num_1x1, num_3x3, num_5x5, num_pool, reduce3, reduce5):
        super(InceptionBlock2a, self).__init__()
        self.conv_cat1 = nn.Conv2d(in_channels, out_channels=num_1x1, kernel_size=1)
        self.conv_cat2pre = nn.Conv2d(in_channels, out_channels=reduce3, kernel_size=1)
        self.conv_cat2 = nn.Conv2d(in_channels=reduce3, out_channels=num_3x3, kernel_size=3, padding=1)
        self.conv_cat3pre = nn.Conv2d(in_channels, out_channels=reduce5, kernel_size=1)
        self.conv_cat3post = nn.Conv2d(in_channels=reduce5, out_channels=num_5x5, kernel_size=3, padding=1)
        self.conv_cat3 = nn.Conv2d(in_channels=num_5x5, out_channels=num_5x5, kernel_size=3,
                                       padding=1)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_cat4 = nn.Conv2d(in_channels=in_channels, out_channels=num_pool, kernel_size=1)

    def forward(self, x_in):
        x1 = F.relu(self.conv_cat1(x_in))
        x2 = F.relu(self.conv_cat2pre(x_in))
        x2 = F.relu(self.conv_cat2(x2))
        x3 = F.relu(self.conv_cat3pre(x_in))
        x3 = F.relu(self.conv_cat3post(x3))
        x3 = F.relu(self.conv_cat3(x3))
        x4 = self.mp(x_in)
        x4 = F.relu(self.conv_cat4(x4))
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class InceptionPool(nn.Module):

    def __init__(self,in_channels, num_3x3, num_5x5, num_pool, reduce3, reduce5):
        super(InceptionPool, self).__init__()
        self.conv_cat1pre = nn.Conv2d(in_channels, out_channels=reduce5, kernel_size=1)
        self.conv_cat1post = nn.Conv2d(in_channels=reduce5, out_channels=num_5x5, kernel_size=3, padding=1)
        self.conv_cat1 = nn.Conv2d(in_channels=num_5x5, out_channels=num_5x5, kernel_size=3, stride=2,
                                       padding=1)
        self.conv_cat2pre = nn.Conv2d(in_channels, out_channels=reduce3, kernel_size=1)
        self.conv_cat2 = nn.Conv2d(in_channels=reduce3, out_channels=num_3x3, kernel_size=3, stride=2,
                                       padding=1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_cat3 = nn.Conv2d(in_channels=in_channels, out_channels=num_pool, kernel_size=1)

    def forward(self, x_in):
        x1 = F.relu(self.conv_cat1pre(x_in))
        x1 = F.relu(self.conv_cat1post(x1))
        x1 = F.relu(self.conv_cat1(x1))
        x2 = F.relu(self.conv_cat2pre(x_in))
        x2 = F.relu(self.conv_cat2(x2))
        x3 = self.mp(x_in)
        x3 = F.relu(self.conv_cat3(x3))

        x = torch.cat((x1, x2, x3), dim=1)

        return x



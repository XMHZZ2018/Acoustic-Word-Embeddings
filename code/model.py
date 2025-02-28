import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        initialization=True,
    ):
        super(BasicBlock, self).__init__()
        self.initialization = initialization
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )

        if initialization:
            self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class GatingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        initialization=True,
        activation=None,
    ):
        super(GatingBlock, self).__init__()

        self.activation = activation

        self.block1 = BasicBlock(
            in_channels, out_channels, kernel_size, stride, padding, initialization
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.block2 = BasicBlock(
            in_channels, out_channels, kernel_size, stride, padding, initialization
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.block1(x)
        out1 = F.sigmoid(self.bn1(out1))
        out2 = self.block2(x)
        out2 = self.bn2(out2)
        if self.activation != None:
            out2 = self.activation(out2)
        return out1 * out2


class LinearGatingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(LinearGatingBlock, self).__init__()

        self.activation = activation

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = F.sigmoid(self.bn1(out1))
        out2 = self.fc2(x)
        out2 = self.bn2(out2)
        if self.activation != None:
            out2 = self.activation(out2)
        return out1 * out2


class GatedCNN(nn.Module):
    def __init__(self, out_dims, initialization=True, activation=None):
        super(GatedCNN, self).__init__()

        self.activation = activation

        self.gb1 = GatingBlock(
            39, 512, 9, initialization=initialization, activation=activation
        )
        self.pool1 = nn.MaxPool1d(3)

        self.bottleneck = nn.Sequential(
            GatingBlock(
                512, 128, 3, padding=1, initialization=False, activation=activation
            ),
            GatingBlock(
                128, 128, 9, padding=4, initialization=False, activation=activation
            ),
            GatingBlock(
                128, 512, 3, padding=1, initialization=False, activation=activation
            ),
        )

        self.pool2 = nn.MaxPool1d(16)
        self.lgb = LinearGatingBlock(2048, 1024, activation=activation)
        self.fc = nn.Linear(1024, out_dims)
        self.softmax = torch.nn.Softmax(dim=1)

        if initialization:
            self.init_weights()

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.gb1(x)
        out = self.pool1(out)
        out = self.bottleneck(out)
        out = self.pool2(out)
        out = out.view(-1, 2048)
        out = self.lgb(out)
        out = self.fc(out)
        return out

    def get_embeds(self, out):
        return self.softmax(out)


class Siamese(nn.Module):
    """
    Triplet Siamese network.
    """

    def __init__(
        self, submod, out_dims, margin=0.4, initialization=True, activation=None
    ):
        """
        submod is the sub-network used in Siamese.
        """
        super(Siamese, self).__init__()

        self.activation = activation
        self.margin = margin
        self.subnet = submod(out_dims, initialization, activation)

    def forward(self, achor, same, diff):
        out_a = self.subnet(achor)
        out_s = self.subnet(same)
        out_d = self.subnet(diff)
        return torch.mean(
            F.relu(
                self.margin
                + F.cosine_similarity(out_a, out_d)
                - F.cosine_similarity(out_a, out_s)
            )
        )

    def get_embeds(self, x):
        return self.subnet(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, activation=None, downsample=None):
        super(Bottleneck, self).__init__()
        self.activation = activation
        self.stride = stride

        self.gb1 = GatingBlock(
            inplanes, planes, 3, padding=1, initialization=False, activation=activation
        )
        self.gb2 = GatingBlock(
            planes,
            planes,
            9,
            stride=stride,
            padding=4,
            initialization=False,
            activation=activation,
        )
        self.gb3 = GatingBlock(
            planes,
            planes * self.expansion,
            3,
            padding=1,
            initialization=False,
            activation=activation,
        )

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.gb1(x)
        out = self.gb2(out)
        out = self.gb3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class biLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nout, ninp, nhid, nlayers, dropout=0.4):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(
            ninp, nhid, nlayers, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.decoder = nn.Linear(nhid * 2, nout)

        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.nout = nout

        self.init_weights()

    def forward(self, input, hidden):
        self.lstm.flatten_parameters()
        output, h = self.lstm(input, hidden)
        decoded = self.decoder(
            h[0][-2:, :, :].transpose(0, 1).contiguous().view(-1, self.nhid * 2)
        )
        return decoded, h

    def init_weights(self):
        initrange = 0.5
        self.lstm.named_parameters()
        for name, val in self.lstm.named_parameters():
            if name.find("bias") == -1:
                # getattr(self.lstm, name).data.uniform_(-initrange, initrange)
                getattr(self.lstm, name).data.normal_(
                    0, math.sqrt(2.0 / (self.ninp + self.nhid))
                )
            else:
                getattr(self.lstm, name).data.fill_(0)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(
            0, math.sqrt(2.0 / (self.nhid * 2 + self.nout))
        )  # bidirectional

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
            Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
        )


class CrossView(nn.Module):
    def __init__(
        self, submod, out_dims, margin=0.4, initialization=True, activation=None
    ):
        super(CrossView, self).__init__()

        self.activation = activation
        self.margin = margin

        # Initialize constituent networks
        self.gc = submod(out_dims, initialization, activation)
        self.lstm = biLSTM(out_dims, 26, 512, 2)

        # Initialize optimizers
        # self.optimizer_gc = optim.Adam(self.gc.parameters())
        # self.optimizer_lstm = optim.Adam(self.lstm.parameters())

    def forward(self, achor, same, diff, hidden, batch_size):
        out_a, hidden = self.lstm(achor, hidden)
        out_s = self.gc(same)
        out_d = self.gc(diff)
        return out_a, out_s, out_d, hidden

    def loss(self, achor, same, diff, hidden, batch_size):
        out_a, out_s, out_d, hidden = self.forward(
            achor, same, diff, hidden, batch_size
        )
        return (
            torch.mean(
                F.relu(
                    self.margin
                    + F.cosine_similarity(out_a, out_d)
                    - F.cosine_similarity(out_a, out_s)
                )
            ),
            hidden,
        )

    def get_embeds(self, x):
        return self.gc(x)

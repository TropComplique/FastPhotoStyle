import torch.nn as nn


class VGGEncoder(nn.Module):

    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, kernel_size=1)

        # PART 1

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # PART 2

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # PART 3

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # PART 4

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3)
        self.relu4_1 = nn.ReLU(inplace=True)

    def forward(self, x, level=4):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            level: an integer, possible values are [1, 2, 3, 4].
        Returns:
            features: a dict with float tensors.
            pooling_indices: a dict with tuples (long tensor, shape).
        """
        features = {}
        pooling_indices = {}

        # image standardization
        x = self.conv0(x)

        x = self.pad1_1(x)
        x = self.conv1_1(x)
        x = self.relu1_1(x)

        features[1] = x

        if level < 2:
            return features, pooling_indices

        x = self.pad1_2(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)

        prepooling_shape = x.size()
        x, indices = self.maxpool1(x)

        x = self.pad2_1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)

        features[2] = x
        pooling_indices[1] = (indices, prepooling_shape)

        if level < 3:
            return features, pooling_indices

        x = self.pad2_2(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)

        prepooling_shape = x.size()
        x, indices = self.maxpool2(x)

        x = self.pad3_1(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)

        features[3] = x
        pooling_indices[2] = (indices, prepooling_shape)

        if level < 4:
            return features, pooling_indices

        x = self.pad3_2(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)

        x = self.pad3_3(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)

        x = self.pad3_4(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)

        prepooling_shape = x.size()
        x, indices = self.maxpool3(x)

        x = self.pad4_1(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)

        features[4] = x
        pooling_indices[3] = (indices, prepooling_shape)

        return features, pooling_indices


class VGGDecoder(nn.Module):

    def __init__(self, level):
        """
        Arguments:
            level: an integer, possible values are [1, 2, 3, 4].
        """
        super(VGGDecoder, self).__init__()
        self.level = level

        if level > 3:

            self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv4_1 = nn.Conv2d(512, 256, kernel_size=3)
            self.relu4_1 = nn.ReLU(inplace=True)

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3)
            self.relu3_4 = nn.ReLU(inplace=True)

            self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3)
            self.relu3_3 = nn.ReLU(inplace=True)

            self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3)
            self.relu3_2 = nn.ReLU(inplace=True)

        if level > 2:

            self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_1 = nn.Conv2d(256, 128, kernel_size=3)
            self.relu3_1 = nn.ReLU(inplace=True)

            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3)
            self.relu2_2 = nn.ReLU(inplace=True)

        if level > 1:

            self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_1 = nn.Conv2d(128, 64, kernel_size=3)
            self.relu2_1 = nn.ReLU(inplace=True)

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3)
            self.relu1_2 = nn.ReLU(inplace=True)

        if level > 0:

            self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_1 = nn.Conv2d(64, 3, kernel_size=3)

    def forward(self, x, pooling_indices):
        """
        Arguments:
            x: a float tensor with shape [b, c, h, w].
            pooling_indices: a dict with tuples (long tensor, shape).
        Returns:
            a float tensors.
        """

        if self.level > 3:

            x = self.pad4_1(x)
            x = self.conv4_1(x)
            x = self.relu4_1(x)

            indices, prepooling_shape = pooling_indices[3]
            x = self.unpool3(x, indices, prepooling_shape)

            x = self.pad3_4(x)
            x = self.conv3_4(x)
            x = self.relu3_4(x)

            x = self.pad3_3(x)
            x = self.conv3_3(x)
            x = self.relu3_3(x)

            x = self.pad3_2(x)
            x = self.conv3_2(x)
            x = self.relu3_2(x)

        if self.level > 2:

            x = self.pad3_1(x)
            x = self.conv3_1(x)
            x = self.relu3_1(x)

            indices, prepooling_shape = pooling_indices[2]
            x = self.unpool2(x, indices, prepooling_shape)

            x = self.pad2_2(x)
            x = self.conv2_2(x)
            x = self.relu2_2(x)

        if self.level > 1:

            x = self.pad2_1(x)
            x = self.conv2_1(x)
            x = self.relu2_1(x)

            indices, prepooling_shape = pooling_indices[1]
            x = self.unpool1(x, indices, prepooling_shape)

            x = self.pad1_2(x)
            x = self.conv1_2(x)
            x = self.relu1_2(x)

        if self.level > 0:

            x = self.pad1_1(x)
            x = self.conv1_1(x)

        return x

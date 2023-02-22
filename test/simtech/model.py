import tensorflow as tf
from tensorflow import keras


def swish(x):
    return x * tf.nn.sigmoid(x)


class SEBlock(keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.reduced_conv = keras.layers.Conv2D(
            filters=self.num_reduced_filters, kernel_size=(1, 1), strides=1, padding="same")
        self.expand_conv = keras.layers.Conv2D(
            filters=input_channels, kernel_size=(1, 1), strides=1, padding="same")

    def call(self, inputs, **kargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduced_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


class MBConv(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_connect_rate = drop_connect_rate
        self.conv1 = keras.layers.Conv2D(
            filters=in_channels * expansion_factor, kernel_size=(1, 1), stride=1, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=k, strides=stride, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.se = SEBlock(in_channels * expansion_factor)
        self.conv2 = keras.layers.Conv2D(
            filters=out_channels, kernel_size=(1, 1), strides=1, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.dropout_connect_rate:
                x = self.dropout(x, training=training)
            x = keras.layers.add([x, inputs])
        return x

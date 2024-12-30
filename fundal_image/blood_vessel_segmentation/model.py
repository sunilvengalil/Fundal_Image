from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape,
               num_output_channels=1,
               num_filters=[64, 128, 256, 512],
               num_channels_z = 1024,
               name="UNET"):
    inputs = Input(input_shape)

    layer_num = 0
    s1, p1 = encoder_block(inputs, num_filters[layer_num])
    layer_num += 1
    s2, p2 = encoder_block(p1, num_filters[layer_num])
    layer_num += 1
    s3, p3 = encoder_block(p2, num_filters[layer_num])
    layer_num += 1
    s4, p4 = encoder_block(p3, num_filters[layer_num])

    b1 = conv_block(p4, num_channels_z)

    d1 = decoder_block(b1, s4, num_filters[layer_num])
    layer_num -= 1
    d2 = decoder_block(d1, s3, num_filters[layer_num])
    layer_num -= 1
    d3 = decoder_block(d2, s2, num_filters[layer_num])
    layer_num -= 1
    d4 = decoder_block(d3, s1, num_filters[layer_num])
    layer_num -= 1

    outputs = Conv2D(num_output_channels, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name=name)
    return model


if __name__ == "__main__":
    test_model = build_unet(input_shape=(512, 512, 3))
    test_model.summary()

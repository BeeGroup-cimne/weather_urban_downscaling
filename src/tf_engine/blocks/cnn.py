from tensorflow.keras.layers import TimeDistributed, Conv2D, BatchNormalization, LeakyReLU, Add

def res_block(x, filters):
    skip = x
    x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(alpha=0.2))(x)
    x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Add()([x, skip])
    return x

def conv_block(x, filters):
    x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(Conv2D(filters, (3, 3), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    return x

import keras
from keras import Sequential, Input, Model, callbacks
from keras.layers import Embedding, Conv1D, MaxPool1D, Concatenate, Dropout, Dense
from keras.utils.np_utils import  to_categorical
import numpy as np
import os.path


data_dir = os.path.abspath('./data')
log_dir = os.path.abspath('./logs')
best_model_path = os.path.abspath('./model/best.h5')
current_model_path = os.path.abspath('./model/current.h5')
csv_log_path = os.path.abspath('./csv_trainig_log.csv')
embedding_mx_width = 128
batch_size = 64
epoch_num = 100


train_data = np.load(os.path.join(data_dir, 'train.npz'))
x_train, y_train = train_data['x'], train_data['y']
one_hot_y_train = to_categorical(y_train)
print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape,
                                                                              one_hot_y_train.shape))

val_data = np.load(os.path.join(data_dir, 'val.npz'))
x_val, y_val = val_data['x'], val_data['y']
one_hot_y_val = to_categorical(y_val)
print(
    "Validation Data loaded with shape: {} and labels with shape - {}".format(x_val.shape, one_hot_y_val.shape))
no_of_classes = one_hot_y_train.shape[1]

model = Sequential()
model.add(Embedding(
    input_dim=256,
    output_dim=embedding_mx_width,
    input_length=(512,),
    input_shape=(512,)
))

kernels= {}
kernels[0] = 4
kernels[1] = 8
kernels[2] = 16
input_shape_conv = (512, 128)
max_pool_size = 2

input = keras.Input(shape=input_shape_conv)
convs = []

for conv_num in range(len(kernels)):
    conv = Conv1D(filters=30,
                  kernel_size=kernels[conv_num],
                  activation='relu',
                  input_shape=input_shape_conv,
                  data_format='channels_last')(input)
    pool = MaxPool1D(pool_size=max_pool_size)(conv)
    convs.append(pool)

out = Concatenate(axis=1)(convs)

conv_model = Model(input=input, output=out)
#conv_model.summary()

model.add(conv_model)
#model.add(keras.layers.Reshape(target_shape=(754,)))
model.add(keras.layers.Reshape(target_shape=(22620,)))
model.add(Dropout(rate=0.5))
model.add(Dense(no_of_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.005)))





callback_list = [
    callbacks.ModelCheckpoint(filepath=current_model_path, verbose=1),
    callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_accuracy', save_best_only=True),
    callbacks.CSVLogger(filename=csv_log_path, append=True),
    callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# model.build(input_shape=(512,1))
model.summary()


model.fit(
    x=x_train,
    y=one_hot_y_train,
    epochs=epoch_num,
    batch_size=batch_size,
    validation_data=(x_val, one_hot_y_val),
    verbose=2,
    callbacks=callback_list
)





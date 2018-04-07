import tensorflow as tf
import numpy as np
import random

# Image manipulation.
import PIL.Image
from PIL import ImageTk
import tkinter as tk

import matplotlib.pyplot as plt
from IPython.display import clear_output



from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
                            BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform

from keras.models import model_from_json
from keras.callbacks import Callback

from keras.preprocessing.image import ImageDataGenerator



import keras
keras.backend.set_image_data_format('channels_last')
keras.backend.set_learning_phase(1)


####################################################################################################

def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.

    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image / 255.0, 0.0, 1.0)

        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        image = PIL.Image.fromarray(image)
        image.show()
        # display(image)


def display(img):
    root = tk.Tk()
    tkimage = ImageTk.PhotoImage(img)
    tk.Label(root, image=tkimage).pack()
    root.mainloop()

def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized


from itertools import compress
class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show();


plot = PlotLearning()

################################################################################################################
                                    #Models#
################################################################################################################

def identity_block(X, f, filters, stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    ##### MAIN PATH #####
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)


    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape, classes):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (11, 11), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), name="avg_pool")(X)


    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def ConvNet(input_shape, classes):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # X_input = Input(input_shape)

    # X = Conv2D(32, (3, 3), strides=(1,1), input_shape=input_shape, activation='relu', name='conv1')(X_input)
    # X = Conv2D(32, (3, 3), strides=(1,1), activation='relu', name='conv2')(X)
    # X = MaxPooling2D(pool_size=(2,2), name='max_pool_1')(X)
    # X = Dropout(0.2)(X)
    #
    # X = Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv3')(X)
    # X = Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv4')(X)
    # X = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(X)
    # X = Dropout(0.2)(X)
    #
    # X = AveragePooling2D(pool_size=(2, 2), name="avg_pool_1")(X)
    #
    # # output layer
    # X = Flatten()(X)
    # X = Dense(128, activation='relu', name='fc1')(X)
    # X = Dropout(0.2)(X)
    # X = Dense(classes, activation='softmax', name='fc2')(X)
    #
    # # Create model
    # model = Model(inputs=X_input, outputs=X, name='ConvNet')(X)

    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), strides=(1,1), input_shape=input_shape, activation='relu', name='conv1'))
    model.add(Conv2D(32, (3, 3), strides=(1,1), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='max_pool_1'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv3'))
    model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool_2'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='fc1'))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax', name='fc2'))

    return model


################################################################################################################
                                    #TensorFlowSession#
################################################################################################################

from pathlib import Path
import glob
import os

seed = 1
np.random.seed(seed)


BATCH_SIZE=128
EPOCHS=200
IMAGE_HEIGHT=240
IMAGE_WIDTH=320


images_root = Path('flowers')
data_dir = list(os.walk(images_root))
classes = data_dir[0][1]
num_classes = len(classes)

X_data = []
Y_data = []

#load the data
for lib in data_dir[1:]:
    lib_class = (lib[0])[8:]
    files = glob.glob (lib[0] + "/*.JPG")
    for myFile in files:
        image = resize_image(load_image(myFile), size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        X_data.append(image)
        Y_data.append(np.array([name==lib_class for name in classes]).astype(int))


X_data, Y_data = np.asarray(X_data), np.asarray(Y_data)

#shuffle and split the data
X_data, Y_data = shuffle_in_unison(X_data, Y_data)
split_idx = int(0.9 * len(X_data))
X_train, X_test = X_data[:split_idx], X_data[split_idx:]
Y_train, Y_test = Y_data[:split_idx], Y_data[split_idx:]



# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    width_shift_range=0.1,
#                                    height_shift_range=0.1,
#                                    horizontal_flip=True,
#                                    vertical_flip=False)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# training_set = train_datagen.flow_from_directory('flowers',
#                                                  target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
#                                                  batch_size=BATCH_SIZE,
#                                                  class_mode='categorical')
#
# test_set = test_datagen.flow_from_directory('dataset/vl',
#                                             target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
#                                             batch_size=BATCH_SIZE,
#                                             class_mode='categorical')


keras.backend.clear_session()

# model execution realm
load_flag = False

if load_flag:
    model = load_model()
else:
    model = ResNet50(input_shape = X_train.shape[1:], classes = num_classes)

model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)


print("FINISHED FITTING")

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# save_model(model)

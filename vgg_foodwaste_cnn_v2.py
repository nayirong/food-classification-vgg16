### Optimise Performance ###
#import tensorflow as tf
#from tensorflow.compat.v1.keras import backend as K
import os

#NUM_PARALLEL_EXEC_UNITS = int(os.cpu_count()/2)
#config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
#        allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
#session = tf.compat.v1.Session(config=config)
#K.set_session(session)
#os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
#os.environ["KMP_BLOCKTIME"] = "30"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

### Import Libraries ###
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import numpy as np

from PIL import Image

### Data Setup ###
# dataset from http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
# labels available at https://github.com/alpapado/food-101/blob/master/data/meta/labels.txt
# each example in generator is tuple in the form ((batch_size, ?, ?, 3), (batch_size, label))
batch_size = 16
num_classes = 35

##Data directory##
training_dir = '/home/yirongyirongyirong/train_full'
test_dir = '/home/yirongyirongyirong/validation'
#len_train_data = num_classes * 1000 * 0.75
#len_validation_data = num_classes * 1000 * 0.25
len_train_data = 33956
len_validation_data = 362

###Data Setup ###
IMAGE_SIZE = [64,64] # we will keep the image size as (64,64). You can increase the size for better results. 

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

# augment data on top of transforming data
datagen1 = ImageDataGenerator(rescale = 1./255,
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen2 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen3 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              zoom_range=[0.2, 1.0],
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen4 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              zoom_range=[0.2, 1.0],
                              height_shift_range=0.25,
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen5 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              zoom_range=[0.2, 1.0],
                              height_shift_range=0.25,
                              width_shift_range=[-56, 56],
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen6 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              zoom_range=[0.2, 1.0],
                              height_shift_range=0.25,
                              width_shift_range=[-56, 56],
                              brightness_range=[0.5, 1.0],
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen7 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              zoom_range=[0.2, 1.0],
                              height_shift_range=0.25,
                              width_shift_range=[-56, 56],
                              brightness_range=[0.5, 1.0],
                              horizontal_flip=True,
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)
datagen8 = ImageDataGenerator(rescale = 1./255,
                              shear_range=0.2,
                              zoom_range=[0.2, 1.0],
                              height_shift_range=0.25,
                              width_shift_range=[-56, 56],
                              brightness_range=[0.5, 1.0],
                              horizontal_flip=True,
                              vertical_flip=True,
                              preprocessing_function=preprocess_input,
                              validation_split=0.2)

train1 = datagen1.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train2 = datagen2.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train3 = datagen3.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train4 = datagen4.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train5 = datagen5.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train6 = datagen6.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train7 = datagen7.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
train8 = datagen8.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='training')
all_train_data = [train1, train2, train3, train4, train5, train6, train7, train8]

validation1 = datagen1.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation2 = datagen2.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation3 = datagen3.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation4 = datagen4.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation5 = datagen5.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation6 = datagen6.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation7 = datagen7.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
validation8 = datagen8.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = batch_size, seed = 11, class_mode='categorical', subset='validation')
all_validation_data = [validation1, validation2, validation3, validation4,
                       validation5, validation6, validation7, validation8]

### Load Model If Previous Versions Exist ###
model_exists = False
latest_model = None

# for filename in os.listdir():
#     if '.h5' in filename and 'vgg-model-35classes' in filename:
#         model_exists = False #True
#         if latest_model != None:    # compare based on validation accuracy
#             file_val_accuracy = filename.split('-')[4][:-3]
#             latest_val_accuracy = latest_model.split('-')[4][:-3]
#             if float(file_val_accuracy) > float(latest_val_accuracy):
#                 latest_model = filename
#             elif float(file_val_accuracy) == float(latest_val_accuracy):    # if same validation accuracy, choose higher training accuracy
#                 file_accuracy = filename.split('-')[3]
#                 latest_accuracy = latest_model.split('-')[3]
#                 if float(file_accuracy) > float(latest_accuracy):
#                     latest_model = filename
#                 elif float(file_accuracy) == float(latest_accuracy):    # choose latest model
#                     file_epoch_no = filename.split('-')[1]
#                     latest_epoch_no = latest_model.split('-')[1]
#                     if int(file_epoch_no) > int(latest_epoch_no):
#                         latest_model = filename
#         else:
#             latest_model = filename

# if model_exists:
#     model = load_model(latest_model)
#     print('Loaded Model: ' + latest_model)
# else:
    # # loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
    # vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

    # # this will exclude the initial layers from training phase as there are already been trained.
    # for layer in vgg.layers:
    #     layer.trainable = False
    
    # model_transfer_half = Sequential()
    # model_transfer_half.add(vgg)
    # model_transfer_half.add(GlobalAveragePooling2D())
    # model_transfer_half.add(Dropout(0.2))
    # model_transfer_half.add(Dense(100, activation='relu'))
    # model_transfer_half.add(Dense(11, activation='softmax'))
    # model_transfer_half.summary()

    # x = Flatten()(vgg.output)
    # #x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
    # x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.
    # model = Model(inputs = vgg.input, outputs = x)

# inception-model for InceptionV3, saved-model for MobileNetV2
#save model at each epoch
# epoch_save = ModelCheckpoint('vgg-model-35classes-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.h5', monitor='loss', verbose=0, save_best_only=False,
#                              save_weights_only=False, mode='auto', period=1)

# optimizer = Adam(learning_rate = 0.002) #default = 0.001
# #print(model.summary())
# model.compile(optimizer=optimizer,#'adam',
#         loss='categorical_crossentropy', metrics=["accuracy"])

# num_epochs = 20

# #for ctr in range(len(all_train_data)):
# history = model.fit(train_data, steps_per_epoch = train_data/(batch_size*num_epochs),
#                         batch_size=batch_size, epochs=num_epochs, validation_data=validation_data,
#                         validation_steps = validation_data/(batch_size*num_epochs,
#                         verbose=1, callbacks=[epoch_save])
# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False
    
model_transfer_half = Sequential()
model_transfer_half.add(vgg)
model_transfer_half.add(GlobalAveragePooling2D())
model_transfer_half.add(Dropout(0.2))
model_transfer_half.add(Dense(100, activation='relu'))
model_transfer_half.add(Dense(11, activation='softmax'))
model_transfer_half.summary()

num_epochs = 20

history = model_transfer_half.fit(all_train_data[ctr], steps_per_epoch = train1, #int(all_train_data[ctr].samples/batch_size),
                        batch_size=batch_size, epochs=num_epochs, validation_data= validation1,#all_validation_data[ctr],
                        validation_steps = int(all_validation_data[ctr].samples/batch_size),
                        verbose=1, callbacks=[epoch_save])
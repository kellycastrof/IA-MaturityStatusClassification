import keras
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Dense, Activation
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.layers import Flatten, Dropout
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import SGD
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
import sys
import pandas as pd



def summarize_diagnostics(history):
    pyplot.title('MobileNet Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='Train')
    pyplot.plot(history.history['val_loss'], color='orange', label='Validation')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('./graficas/Loss_MB.png')
    pyplot.figure()

    pyplot.title('MobileNet Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='Train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='Validation')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Accuracy')
    
    pyplot.legend()
    pyplot.savefig('./graficas/Accuracy_MB.png')
    pyplot.figure()
    pyplot.close()

base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3)) 
#Transfer Learning
for layer in base_model.layers:
    layer.trainable=False

x=base_model.output
x = Flatten()(base_model.layers[-1].output)
x=Dense(1024,activation='relu')(x) #dense layer 1
x = Dropout(0.20)(x)
x=Dense(512,activation='relu')(x) #dense layer 2
x = Dropout(0.20)(x)
x=Dense(256,activation='relu')(x) #dense layer 3
x = Dropout(0.20)(x)
preds=Dense(4,activation='softmax')(x) #final layer - softmax activation
model=Model(inputs=base_model.input,outputs=preds)
model.summary()


model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


datagenerator=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=datagenerator.flow_from_directory('./datasetreal/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

validation_generator=datagenerator.flow_from_directory('./datasetreal/validation',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

test_generator= datagenerator.flow_from_directory('./datasetreal/test',
                                                target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False
                                                )





history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=len(validation_generator), epochs=30, verbose=1)
model.summary()

_, acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=0)
print('> %.3f' % (acc * 100.0))

hist_df = pd.DataFrame(history.history)
hist_csv_file = './graficas/historyMB.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

summarize_diagnostics(history)

model.save('./modelo/modeloMB.h5')
model.save_weights('./modelo/pesosMB.h5')


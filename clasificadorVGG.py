import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import SGD
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint


import sys

# mostrar las curvas de diagnóstico del aprendizaje
def summarize_diagnostics(history):
    # plot error
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot exactitud
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # guardar el plot en un archivo ... por si lo queremos publicar
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

base_model=VGG16(weights='imagenet',include_top=False,input_shape=(224, 224, 3)) 
#Transfer Learning
for layer in base_model.layers:
    layer.trainable=False

x=base_model.output
x = Flatten()(base_model.layers[-1].output)
x=Dense(128,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(128,activation='relu')(x) #dense layer 2
x=Dense(64,activation='relu')(x) #dense layer 2
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)
model.summary()
# compilar el modelo
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


dategenerator=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=dategenerator.flow_from_directory('./datasetreal/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=50,
                                                 class_mode='categorical',
                                                 shuffle=False)

validation_generator=dategenerator.flow_from_directory('./datasetreal/validation',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=50,
                                                 class_mode='categorical',
                                                 shuffle=False)


model.compile(loss='categorical_crossentropy',metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='bestVGG.hdf5', verbose=1, save_best_only=True)
history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=len(validation_generator), epochs=50, verbose=1)
model.summary()

# curvas de de aprendizaje
summarize_diagnostics(history)

model.save('modelo.h5')
model.save_weights('pesos.h5')

 #neuronas en la capa de salida de la cnn para la 1era capa oculta
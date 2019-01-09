import numpy as np
import itertools
import os
import pickle
from glob import glob
import tensorflow
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD, RMSprop
from keras.metrics import top_k_categorical_accuracy
from PIL import ImageFile
import os


# from Adam_lr_mult import Adam_lr_mult
# from ModelMGPU import ModelMGPU

def create_learn_rate_dict(model):
    """
    Since we're using a custom optimizer with a different learning rate for each layer, we need to initialize a dictionary of layer names and weights. Here I'm using one lower value for all but the last layer
    """
    base_layer_learn_ratio = 0.1
    final_layer_learn_ratio = 1
    layer_mult = dict(zip([layer.name for layer in model.layers],
                          itertools.repeat(base_layer_learn_ratio)))
    layer_mult[model.layers[-1].name] = final_layer_learn_ratio
    return layer_mult

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def get_fresh_xception(img_size, n_categories, weights='imagenet'):
    base_model = Xception(weights='imagenet',
                    include_top=False,
                    input_shape=img_size)

    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    predictions = Dense(n_categories, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=predictions)
    lr_dict = create_learn_rate_dict(model)
    Adam_lr_optimizer = Adam_lr_mult(multipliers=lr_dict)
    model.compile(optimizer=Adam_lr_optimizer, loss='categorical_crossentropy', metrics=[top_3_accuracy])
    return model


class XceptionDataGenCreator():
    def __init__(self,target_size=(299,299),batch_size=4):
        self.target_size = target_size
        self.batch_size = batch_size

    def get_train_gen(self, train_path):
        gen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                 rotation_range=40,
                                 width_shift_range=0.1,
                                 height_shift_range=0.2,
                                 shear_range=0.1,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest'
                                 )
        self.train_gen = gen.flow_from_directory(train_path,
                                            target_size=self.target_size,
                                            batch_size=self.batch_size)
        return self.train_gen

    def get_val_gen(self, val_path):
        gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.val_gen = gen.flow_from_directory(val_path,
                                            target_size=self.target_size,
                                            batch_size=self.batch_size) 
        return self.val_gen

    def get_test_gen(self, test_path):
        gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_gen = gen.flow_from_directory(test_path,
                                            target_size=self.target_size,
                                            batch_size=self.batch_size)
        return self.test_gen


if __name__ == '__main__':




    train_path = '/Users/josephshull/Dropbox/fish_photos/data/train'
    validation_path = '/Users/josephshull/Dropbox/fish_photos/data/validation'
    holdout_path = '/Users/josephshull/Dropbox/fish_photos/data/test'
    
    img_size = (299,299,3)
    target_size = (img_size[0],img_size[1])
    batch_size = 16
    steps_per_epoch = 19445//batch_size
    num_epochs = 100
    validation_steps = 2513//batch_size
    CPUS = 6
    GPUS = 1


    # Make DataGen
    gencreator = XceptionDataGenCreator(target_size=target_size, batch_size=batch_size)
    train_generator = gencreator.get_train_gen(train_path)
    validation_generator = gencreator.get_val_gen(validation_path)
    holdout_generator = gencreator.get_test_gen(holdout_path)

    #save classes for decoding
    class_dict = train_generator.class_indices
    class_decode = dict([(value,key) for (key,value) in class_dict.items()])
    np.save('decode_class.npy', class_decode)

    #Make the model
    n_categories = len(os.listdir(train_path))
    model = get_fresh_xception(img_size, n_categories)

    #Activate GPUS
    if GPUS >2:
        model = ModelMGPU(model)


    #Get the callbacks in order
    tbCallBack = TensorBoard(log_dir='../logs',
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)


    headsavecheckpoint = ModelCheckpoint(filepath='models/Xception-fish.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    #Train it.
    history = model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[tbCallBack, headsavecheckpoint],
                    workers=CPUS,
                    use_multiprocessing=True)

    np.save('train_history.npy', history.history)
    model.save('Xception-fish-end.hdf5')

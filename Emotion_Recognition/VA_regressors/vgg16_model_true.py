from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Dense, Dropout,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
import keras

def get_model():
    IMG_SIZE = 224
    IMG_DIM = 3
    model_base = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, IMG_DIM))
    # print(model.summary())
    # Keeping the weights from the trained network.
    for layers in model_base.layers:
        layers.trainable = False
    ## adding the regression layer on top
    bottom_model = model_base.output
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    bottom_model = Flatten(name="flatten")(bottom_model)
    bottom_model = Dense(4096, activation="relu")(bottom_model)
    # bottom_model = Dense(1024,activation='relu')(bottom_model)
    bottom_model = Dropout(0.50)(bottom_model)
    bottom_model = Dense(4096, activation="relu")(bottom_model)
    # bottom_model = Dense(1024,activation='relu')(bottom_model)
    # bottom_model = Dropout(0.50)(bottom_model)
    # bottom_model = Dense(512,activation='relu')(bottom_model)
    bottom_model = Dropout(0.50)(bottom_model)
    bottom_model = Dense(2, activation='sigmoid')(bottom_model)

    # Connecting the two models
    model = Model(inputs=model_base.input, outputs=bottom_model)
    keras.backend.set_epsilon(1)  # Used for mape
    opt = Adam(lr=0.0001, decay=1e-6)
    # Selecting loss and metrics
    model.compile(optimizer=opt, loss='MSE', metrics=['mse', 'mape'])
    print(model.summary())
    print('completed')

    return model

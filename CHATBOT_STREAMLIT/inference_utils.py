import pickle
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from IPython.display import SVG
import matplotlib.pyplot as plt
import os, re, sys, random, shutil, cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_inception_resnetv2_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained InceptionResNetV2 Model """
    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    # s1 = encoder.get_layer("input_1").output           ## (512 x 512)
    s1 = encoder.get_layer("input_layer").output


    s2 = encoder.get_layer("activation").output        ## (255 x 255)
    s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)         ## (256 x 256)

    s3 = encoder.get_layer("activation_3").output      ## (126 x 126)
    s3 = ZeroPadding2D((1, 1))(s3)                     ## (128 x 128)

    s4 = encoder.get_layer("activation_74").output      ## (61 x 61)
    s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)           ## (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("activation_161").output     ## (30 x 30)
    b1 = ZeroPadding2D((1, 1))(b1)                      ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)
    
    """ Output """
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(6, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs, name="InceptionResNetV2-UNet")
    return model


K.clear_session()

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
model.compile(optimizer=Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])
model.summary()

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.0001, 60)

lr_scheduler = LearningRateScheduler(
    exponential_decay_fn,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath = 'InceptionResNetV2-UNet.h5',
    save_best_only = True, 
#     save_weights_only = False,
    monitor = 'val_loss', 
    mode = 'auto', 
    verbose = 1
)

earlystop = EarlyStopping(
    monitor = 'val_loss', 
    min_delta = 0.001, 
    patience = 12, 
    mode = 'auto', 
    verbose = 1,
    restore_best_weights = True
)

csvlogger = CSVLogger(
    filename= "model_training.csv",
    separator = ",",
    append = False
)

callbacks = [checkpoint, earlystop, csvlogger, lr_scheduler]


model.load_weights(r"C:\Users\sanja\Sem-6\Applied AI\PROJECT\InceptionResNetV2-UNet.h5")
print("Model Loaded Successfully!!!")
# print(model.summary())

id2code = {0: (60, 16, 152),
 1: (132, 41, 246),
 2: (110, 193, 228),
 3: (254, 221, 58),
 4: (226, 169, 41),
 5: (155, 155, 155)}


id2name = {0: 'building',
 1: 'land',
 2: 'road',
 3: 'vegetation',
 4: 'water',
 5: 'unlabeled'}


def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def load_and_preprocess_image(img_path, target_size):
    """
    Load and preprocess the image from a given path.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize if your model expects [0,1] input
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array, img

def predict_mask_on_image(img_array, model):
    """
    Predict the mask from the preprocessed image array.
    """
    pred = model.predict(img_array)
    pred_mask_softmax = pred[0]  # remove batch dimension
    return pred_mask_softmax

def plot_input_and_prediction(user_img, pred_mask_softmax, id2code, save_path='./predictions/user_prediction.png'):
    """
    Plot and save the input user image and predicted mask.
    """
    fig = plt.figure(figsize=(15, 6))

    # Input Image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(user_img)
    ax1.set_title('Input Image', fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax1.grid(False)

    # Predicted Mask
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(onehot_to_rgb(pred_mask_softmax, id2code))
    ax2.set_title('Predicted Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax2.grid(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor='w', bbox_inches='tight', dpi=200)
    plt.show()
    print(f"\n✅ Saved user prediction visualization to '{save_path}'")

def calculate_predicted_area(pred_mask_softmax, id2name):
    """
    Calculate pixel counts and area percentages for prediction only.
    """
    pred_mask = np.argmax(pred_mask_softmax, axis=-1)

    num_classes = pred_mask_softmax.shape[-1]
    total_pixels = np.prod(pred_mask.shape)

    pred_counts = np.bincount(pred_mask.flatten(), minlength=num_classes)

    area_pred = {}

    print(f"\n--- Pixel Statistics for User Image Prediction ---")
    for cls in range(num_classes):
        cls_name = id2name.get(cls, f"Class {cls}")
        print(f"  {cls_name}: Predicted Pixels = {pred_counts[cls]}")

    print(f"\nArea contribution (Predicted):")
    for cls in range(num_classes):
        cls_name = id2name.get(cls, f"Class {cls}")
        pixels = pred_counts[cls]
        percent = (pixels / total_pixels) * 100
        area_pred[cls_name] = {"pixels": int(pixels), "percent": round(percent, 2)}
        print(f"  {cls_name}: {percent:.2f}%")

    return area_pred

def save_user_area_stats(area_pred, save_path='./area_jsons/user_image_area.json'):
    """
    Save the user image's predicted area info to a JSON.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    out_data = {
        "prediction": area_pred
    }

    with open(save_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"\n✅ Saved predicted area info to '{save_path}'")

def run_user_image_inference(img_path, model, id2code, id2name, target_size=(512, 512)):
    """
    Full pipeline to run prediction on a user-provided image.
    Returns predicted area dictionary.
    """
    img_array, user_img = load_and_preprocess_image(img_path, target_size)
    pred_mask_softmax = predict_mask_on_image(img_array, model)
    plot_input_and_prediction(user_img, pred_mask_softmax, id2code)
    area_pred = calculate_predicted_area(pred_mask_softmax, id2name)
    save_user_area_stats(area_pred)
    return area_pred
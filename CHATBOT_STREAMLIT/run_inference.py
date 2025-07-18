from segmentation_utils import (
    load_and_preprocess_image,
    onehot_to_rgb,
    predict_mask,
    plot_prediction,
    calculate_area_distribution,
    save_area_to_json,
)

from model import (build_inception_resnetv2_unet)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    return (2. * tf.reduce_sum(y_true * y_pred) + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.)

# @st.cache_resource
def load_segmentation_model(weights_path=r"C:\Users\sanja\Sem-6\Applied AI\PROJECT\InceptionResNetV2-UNet_BEST.h5"):
    # return tf.keras.models.load_model(weights_path, custom_objects={"dice_coef": dice_coef})
    model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
    model.compile(optimizer=Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])
    # model.summary()
    model.load_weights(weights_path)
    return model

def run_segmentation_pipeline(image_path, model, save_vis=False, save_json=False):
    img_array, img = load_and_preprocess_image(image_path)
    mask_softmax = predict_mask(img_array, model)
    rgb_mask = onehot_to_rgb(mask_softmax)

    area_stats = calculate_area_distribution(mask_softmax)

    if save_vis:
        plot_prediction(img, rgb_mask, save_path='./predictions/user_pred.png')

    if save_json:
        save_area_to_json(area_stats)

    return rgb_mask, area_stats

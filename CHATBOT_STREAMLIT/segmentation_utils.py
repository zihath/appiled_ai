import os, json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Class mapping
ID2CODE = {
    0: (60, 16, 152),
    1: (132, 41, 246),
    2: (110, 193, 228),
    3: (254, 221, 58),
    4: (226, 169, 41),
    5: (155, 155, 155),
}
ID2NAME = {
    0: 'building', 1: 'land', 2: 'road',
    3: 'vegetation', 4: 'water', 5: 'unlabeled'
}


def load_and_preprocess_image(img_path, target_size=(512, 512)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img


def onehot_to_rgb(onehot, colormap=ID2CODE):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap:
        output[single_layer == k] = colormap[k]
    return np.uint8(output)


def predict_mask(img_array, model):
    return model.predict(img_array)[0]  # remove batch dimension


def plot_prediction(img, mask_rgb, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(mask_rgb)
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def calculate_area_distribution(mask_softmax, id2name=ID2NAME):
    mask = np.argmax(mask_softmax, axis=-1)
    total = mask.size
    counts = np.bincount(mask.flatten(), minlength=len(id2name))
    
    return {
        id2name[i]: {
            "pixels": int(counts[i]),
            "percent": round(100 * counts[i] / total, 2)
        } for i in range(len(id2name))
    }


def save_area_to_json(area_dict, save_path='./area_jsons/user_area.json'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(area_dict, f, indent=2)

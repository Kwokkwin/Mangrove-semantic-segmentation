from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

#global PATH variables
abs_path = os.path.abspath("__file__"+"/../../Mangrove-semantic-segmentation/data_set/")
TRAIN_IMAGE_PATH =  os.path.join(abs_path, "train/images/")
TRAIN_LABEL_PATH = os.path.join(abs_path, "train/labels/")
VALIDATION_IMAGE_PATH = os.path.join(abs_path,'validation/images/')
VALIDATION_LABEL_PATH = os.path.join(abs_path, 'validation/labels/')
TEST_IMAGE_PATH = os.path.join(abs_path,'test/images/')
TEST_LABEL_PATH =os.path.join(abs_path, 'test/labels/')
IMAGES_WITHOUT_LABEL_PATH =os.path.join(abs_path, 'images_without_label/')
id2code= {1:(220,245,255),2:(180,245,255),3:(130,245,255),4:(56,245,255),5:(163,255,115),6:(85,255,0),7:(215,243,172),8:(85,127,0),9:(144,173,49),10:(161,255,167),11:(151,230,1),12:(56,168,2),13:(85,212,89),14:(26,197,30),15:(93,255,133),16:(1,230,169),17:(93,255,207),18:(109,190,174),19:(85,190,174),20:(107,191,199),21:(178,178,178),22:(255,0,0),23:(254,234,190),24:(214,133,137),25:(255,127,127),26:(3,197,255),27:(3,181,245),28:(3,167,255),29:(3,137,255),30:(3,107,255),31:(3,77,255),32:(3,47,255),33:(3,17,255),34:(0,77,138),35:(0,77,168),36:(0,77,198),37:(0,77,238),38:(255,0,197),39:(236,96,167),40:(255,125,197),41:(185,142,0),42:(180,171,93),43:(0,0,0)}

# #for grey
# id2code= {1:(1,1,1),2:(2,2,2),3:(3,3,3),4:(4,4,4),5:(5,5,5),6:(6,6,6),7:(7,7,7),8:(8,8,8),9:(9,9,9),10:(10,10,10),11:(11,11,11),12:(12,12,12),13:(13,13,13),14:(14,14,14),15:(15,15,15),16:(16,16,16),17:(17,17,17),18:(18,18,18),19:(19,19,19),20:(20,20,20),21:(21,21,21),22:(,22,,22,,22),23:(23,23,23),24:(24,24,24),25:(25,25,25),26:(26,26,26),27:(27,27,27),28:(28,28,28),29:(29,29,29),30:(30,30,30),31:(31,31,31),32:(32,32,32),33:(33,33,33),34:(34,34,34),35:(35,35,35),36:(36,36,36),37:(37,37,37),38:(38,38,38),39:(39,39,39),40:(40,40,40),41:(41,41,41),42:(42,42,42),43:(43,43,43)}


def rgb_to_onehot(rgb_image, colormap=id2code):
    """
    Function to one hot encode RGB mask labels.
    Args:
        rgb_image: image matrix (eg. 256 x 256 x 3 dimension numpy ndarray).
        colormap: dictionary of color to original_label id.
    Return:
        encoded_image: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap).
    Raises:
        None.
    """
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == colormap[cls], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap=id2code):
    """
    Function to decode encoded mask labels.
    Args:
        onehot: one hot encoded image matrix (height x width x num_classes).
        colormap: dictionary of color to original_label id.
    Return:
        Decoded RGB image (height x width x 3).
    Raises:
        None.
    """
    single_layer = np.argmax(onehot, axis=-1)# get the one with '1'
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]
    return np.uint8(output)


# Define the generator
# Normalizing only frame images, since masks contain original_label info
data_gen_args = dict(rescale=1. / 255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)
test_frames_datagen = ImageDataGenerator(**data_gen_args)
test_masks_datagen = ImageDataGenerator(**mask_gen_args)
images_without_label_datagen = ImageDataGenerator(**data_gen_args)
# Seed defined for aligning images and their masks
seed = 1


def train_data_generator(seed=18, batch_size=8, target_size=(256, 256)):
    """
    training image data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3), mask.
    Raises:
        None.
    """
    train_image_generator = train_frames_datagen.flow_from_directory(
        TRAIN_IMAGE_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size, color_mode='rgb')
    train_mask_generator = train_masks_datagen.flow_from_directory(
        TRAIN_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size, color_mode='rgb')
    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def validation_data_generator(seed=1, batch_size=8, target_size=(256,256)):
    """
    Validation image data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3), mask.
    Raises:
        None.
    """
    val_image_generator = val_frames_datagen.flow_from_directory(
        VALIDATION_IMAGE_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    val_mask_generator = val_masks_datagen.flow_from_directory(
        VALIDATION_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def test_data_generator(seed=1, batch_size=8, target_size=(256, 256)):
    """
    Test image data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3), mask.
    Raises:
        None.
    """
    test_image_generator = test_frames_datagen.flow_from_directory(
        TEST_IMAGE_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    test_mask_generator = test_masks_datagen.flow_from_directory(
        TEST_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    while True:
        X1i = test_image_generator.next()
        X2i = test_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def images_without_label_data_generator(seed=1, batch_size=8, target_size=(256, 256)):
    """
    Images without original_label data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3).
    Raises:
        None.
    """
    images_without_label_generator = images_without_label_datagen.flow_from_directory(
        IMAGES_WITHOUT_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    while True:
        X1i = images_without_label_generator.next()
        yield X1i[0]

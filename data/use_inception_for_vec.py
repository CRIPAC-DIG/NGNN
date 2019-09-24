from PIL import Image
import tensorflow as tf
import numpy as np
import os
import json


"""
************ATTENTION! *****************
the generated image vectors of all clothings are queit big, so please set the suitable directory for saving them.
"""
image = tf.keras.preprocessing.image
preprocess = tf.keras.applications.inception_v3.preprocess_input
myinception = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    pooling='max'
)
file_path = "./images/"
outfit_list = os.listdir(file_path)

for idx, outfit_id in enumerate(outfit_list):
    file_path_outfit = file_path + outfit_id + '/'
    item_list = os.listdir(file_path_outfit)
    print (idx, len(outfit_list))
    for item_id in item_list:
        file_path_outfit_item = file_path_outfit + item_id + '/'

        img = Image.open(file_path_outfit_item)
        img = img.resize((229, 229))
        mat = image.img_to_array(img)
        mat = np.expand_dims(mat, axis=0)
        aa = preprocess(mat)

        itemvector = myinception.predict(aa)

        vector_name = outfit_id + '_' + item_id + '.json'
        with open('./polyvore_image_vectors/' + vector_name, 'w') as f:
            f.write(json.dumps(list(itemvector[0])))



import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json

batch_size = 32
image_size = 224


parser = argparse.ArgumentParser(description='Image predictor app')
#arg1
parser.add_argument('-img','--image_path', type=str, metavar='',required = True, help='image file path')
#arg2
parser.add_argument('-m','--model', type=str, metavar='',required = True, help='trained model name')
parser.add_argument('-k','--top_k', type=int, metavar='',required = True, help='top k')
parser.add_argument('-c','--category_names') 
args = parser.parse_args()
print(args)

print('image_path:', args.image_path)
print('model:', args.model)
print('top_k:', args.top_k)
print('category_names:', args.category_names)

    
class_names = {}
image_shape = (224, 224)

def process_image(img):
    return tf.image.resize(np.squeeze(img), (image_size, image_size))/255.0

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    expanded_image = np.expand_dims(processed_test_image, axis=0)
    prediction = model.predict(expanded_image)
    values, indices = tf.math.top_k(input=prediction, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0]
    return probs, classes, processed_test_image

if __name__ == '__main__':
    print('predict.py, running')
    
    image_path = args.image_path
    
    model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    image, probs, classes = predict(image_path, model, top_k)
    
    print('Propabilties:',probs)
    print('Classes Keys:', classes)
    
    probs, classes, processed_test_image = predict(image_path, model, top_k)
    print('Propabilties:\n', probs)
    print('Classes Keys:', classes)
    flower_classes = []
    print("Classes Values:")
    for i in classes:
        print("-",class_names[str(i+1)])
        flower_classes.append(class_names[str(i+1)])
    fig, (ax1, ax2) = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
    ax1.imshow(processed_test_image)
    ax1.set_title(flower_classes[0])
    #ax2.barh(classes, probs)
    ax2.barh(np.arange(1,103,102/5), probs, 12)
    ax2.set_xticks(np.arange(0,1.1,0.2))
    ax2.set_xlim(0, 1.1)
    ax2.set_yticklabels(flower_classes)
    ax2.set_yticks(np.arange(1,103,102/5))
    ax2.set_ylim(0, 102)
    plt.tight_layout()
    plt.show()

# coding: utf-8

# This is a **Keras** implementation of an **image classifier** following this [blog post](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).
# <br>
# The data comes from the following [Kaggle challenge](https://www.kaggle.com/c/dogs-vs-cats/data)

# Notice that you will need to install the requirements in the `requirements.txt` file by running: <br>
#     `pip install -r requirements.txt`

# In[2]:

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
import numpy as np
import h5py


# ## Images preprocessing

# In[3]:

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# ### An example of an image transformation for data augmentation

# In[4]:

img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
img


# In[7]:

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# In[9]:

transformed_img = load_img('preview/cat_0_1558.jpeg')  # this is a transformed cat image
transformed_img


# ## Building the model

# In[10]:

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# ## Compiling the model
# 

# In[11]:

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# ## Train and test pipelines

# In[12]:

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')


# ## Fit the model

# In[13]:

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)


# In[15]:

model.save_weights('first_try.h5', overwrite=True)  # always save your weights after training or during training


# ## Use the fitted model

# ### Save the model architecture 

# In[19]:

saved_model = model.to_json()
import json
with open('model_architecture.json', 'w') as outfile:
    json.dump(saved_model, outfile)


# ### Load the model (if not already)

# In[28]:

from keras.models import model_from_json
loaded_model = model_from_json(json.load(open('model_architecture.json', 'r')))


# In[16]:

model.summary()


# ### Make some predictions

# In[44]:

train_score = model.evaluate_generator(train_generator, 100)
validation_score = model.evaluate_generator(validation_generator, 100)


# In[45]:

train_score, validation_score


# In[160]:

def load_images(_class='cats'):
    images = np.ndarray(shape=(32, 3, 150, 150))
    for index, img_index in enumerate(range(1000, 1032)):
        if _class == 'cats':
            base_path = 'data/validation/cats/cat.'
        else:
            base_path = 'data/validation/dogs/dog.'
        image = img_to_array(load_img(base_path + str(img_index) + '.jpg'))
        try:
            images[index] = image[:, :, :] / 255
        except ValueError:
            pass
    return images


# In[161]:

test_cat_images = load_images()
test_dog_images = load_images(_class='dogs')


# In[169]:

(img_to_array(load_img('data/validation/cats/cat.1003.jpg')) / 255).shape


# In[157]:

cat_predictions = model.predict(test_cat_images)
dog_predictions = model.predict(test_dog_images)


# In[89]:

model.predict_generator(validation_generator, 32)

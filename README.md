# Image Classification - Multiclass Classification Between Animals

## Description
An image classification prototype between animals which are trained using VGG16 with Transfer learning as its training model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
    - [Structure](#structure)
    - [Developer](#developer)
    - [User](#user)
- [Credits](#credits)

## Installation
1. You might have to install the following dependencies and setup before running the code:
- tensorflow
- opencv-python
- matplotlib
- keras
- numpy
- pillow

2. You can install them in Anaconda Prompt, Anaconda Navigator, or Jupyter Notebook.
- Anaconda Prompt
    - Search for Anaconda Prompt on your device, then run it as an administrator. 
    
    ![Anaconda Prompt](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture1.png)

    - Type ```pip install``` (type the packages you need), then press Enter. 
    
    ![Type pip install](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture2.png)

- Anaconda Navigator
    - Search for Anaconda Navigator. 
    
    ![Anaconda Navigator](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture3.png)

    - Click on Environments. 
    
    ![Click on Environments](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture4.png)

    - Search for the packages that you want to install on the top right-hand side. 
    
    ![Search for the packages](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture5.png)

    - If the packages have been installed, you will see them on the list; otherwise, you need to sort the data to 'Uninstalled' or 'All', search for the package(s), and click on it to install. The installation will take time (mostly depending on the device itself), so it would be better to use the Anaconda Prompt.

- Jupyter Notebook
    - Search for Jupyter Notebook on your device. 
    
    ![Jupyter Notebook](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture6.png)

    - Once directed, type in ```!pip install``` (type the packages you need). 
    
    ![Type !pip install](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture7.png)

    - The installation will take time so all you have to do is to wait until the results appear.
    - You can check whether the packages have been installed by typing ```!pip list```.

    ![Check installed packages](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture8.png)

## Usage
### Structure
Please see the following folder structure of the project:

![Folder Structure](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture9.png)

### Developer
Once all installation has been completed, download the files required:
- image_classification.ipynb (Python file)
- user_interface.ipynb (Python file)
- animal_model.h5 (trained model)

To run the code, extract the zip file and run the image_classification.ipynb file. You will see the following page:

![image_classification.ipynb](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture10.png)

Run through all the instructions in the file (they are also written below).

1. Download and extract the dataset example.

Dataset used for this project: https://www.kaggle.com/datasets/iluvchicken/cheetah-jaguar-and-tiger/versions/1 (You can use your own dataset; do not forget to change the file path of the dataset).

2. Create two folders called 'images' and 'train-test'.

``` python
import os

# dataset directory and extension
parent_dir = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\\"
subdirs = ['images', 'train-test']

for subdir in subdirs:
    newdir = parent_dir + subdir
    
    # create folders
    try:
        os.makedirs(newdir)
        print('Folder ' + subdir + ' created!')
        
    # print message where folders already exist
    except FileExistsError:
        pass
        print('Folder ' + subdir + ' already exist!')
```

You will see the following message:

![Folders created](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture11.png)

3. Copy the downloaded dataset to the 'images' folder.

```python
import os
import shutil

# downloaded dataset directory
parent_dir = r"C:\Users\admir\Downloads\archive\\"

# checking through all images
for folder in os.listdir(parent_dir):
    # dataset source
    folder_path = os.path.join(parent_dir, folder)
    
    # destination
    destination = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\images\\" + folder
    
    # copy from source to destination
    try:
        shutil.copytree(folder_path, destination)
        
    # print message where folders already exist
    except FileExistsError:
        shutil.rmtree(destination)
        shutil.copytree(folder_path, destination)
```

4. Check the dataset.

Since datasets from Kaggle are usually set up properly, there is no need to check the dataset. If you downloaded your own dataset from other websites which are not set up properly (e.g. includes other image extensions), you need to check your dataset first.

```python
import os
import cv2
import imghdr

# image directory and image extension
image_dir = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\images\\"
image_ext = ['jpeg', 'jpg', 'png']

## Remove images
print("Removed images: ")

# checking through all folders
for folder in os.listdir(image_dir):
    remove_count = 0
    
    # checking through all images
    for image in os.listdir(os.path.join(image_dir, folder)):
        image_path = os.path.join(image_dir, folder, image)
        
        # read through image extensions
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            
            # remove image if not included as part of the image extension
            if tip not in image_ext:
                os.remove(image_path)
                remove_count += 1
                                
        except Exception as e:
            print('Issue with image {}'.format(image_path))
      
    # print the number of removed images for each class
    print(folder + ": " + str(remove_count))
```

The number of removed images will be displayed:

![Removed images](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture12.png)

5. Rename files

```python
import os
import cv2
from PIL import Image

# image directory
image_dir = r'C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\images\\'

print("Current total images: ")

# checking through all folder
for folder in os.listdir(image_dir):
    image_count = 0
    
    # checking though all images
    for image in os.listdir(os.path.join(image_dir, folder)):
        # image source
        image_path = os.path.join(image_dir, folder, image)
        image_count += 1
        
        # new image name
        filename = folder + str(image_count) + ".jpg"      
        
        # rename images
        source = image_path
        destination = image_dir + folder + "/" + filename
        os.rename(source, destination)
        
    # print current total images for each class
    print(folder + ": " + str(image_count))
```

The number of the current total images will be displayed:

![Current total images](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture13.png)

6. Create two folders called 'train' and 'test'.

```python
import os

# parent directory and extension
parent_dir = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\train-test\\"
subdirs = ['train', 'test']

for subdir in subdirs:
    newdir = parent_dir + subdir
    
    # create train and test folders
    try:
        os.makedirs(newdir)
        print('Folder ' + subdir + ' created!')
      
    # print message if folders already exist
    except FileExistsError:
        pass
        print('Folder ' + subdir + ' already exists!')
```

You will see the following message:

![Folders created](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture14.png)

7. Create label folders inside the 'train' and 'test' folders.

```python
import os
import shutil
from random import seed
from random import random

# parent directory
parent_dir = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\\"

# label and subdirectory lists
labels = []
subdirs = []

# append the list of labels
for folder in os.listdir(parent_dir + 'images'):
    labels.append(folder)

# create the label folders inside the 'train' and 'test' folders
for folder in os.listdir(parent_dir + 'train-test'):
    subdirs.append(folder)
    
    for subdir in subdirs:
        for label in labels:
            os.makedirs(os.path.join(parent_dir + 'train-test/' + subdir + '/' + label), exist_ok = True)
```

8. Copy 80% of the images to the folders inside the 'train' folder and 20% of the images to the folders inside the 'test' folder randomly.

```python
import os
import shutil
from random import seed
from random import random

# parent directory and subdirectory list
parent_dir = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\\"
subdirs = ['train/', 'test/']

# random and value ratio
seed(1)
val_ratio = 0.2

for folder in os.listdir(parent_dir + 'images'):
    for image in os.listdir(os.path.join(parent_dir + 'images/', folder)):
        # image source
        image_path = os.path.join(parent_dir + 'images/', folder, image)
        
        # destination = subdir[0] ('train' folder)
        destination = parent_dir + 'train-test/' + subdirs[0] + '/' + folder + '/' + image

        # random less than the val_ratio
        if random() < val_ratio:
            # destination = subdir[1] ('test' folder)
            destination = parent_dir + 'train-test/' + subdirs[1] + '/' + folder + '/' + image

        # copy images from source to destination
        shutil.copy(image_path, destination)
```

9. Train the images using VGG16 (Transfer Learning) and save the training model.

```python
# VGG16 Model for Multiclass Image Classification
import os
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# calculate folder
def calculate_folder():
    parent_dir = r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\train-test\test\\"
    folder_count = 0
    
    for folder in os.listdir(parent_dir):
        folder_count += 1
    
    return folder_count

# define cnn model
def define_model():
    # load model
    model = VGG16(include_top = False, input_shape = (224, 224, 3))
    
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu', kernel_initializer = 'he_uniform')(flat1)
    
    # softmax are usually used for multiclass classification
    output = Dense(calculate_folder(), activation = 'softmax')(class1)
    
    # define new model
    model = Model(inputs = model.inputs, outputs = output)
    
    # compile model
    opt = SGD(learning_rate = 0.001, momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

# plot diagnostic learning curves
def summarise_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color = 'red', label = 'train')
    pyplot.plot(history.history['val_loss'], color = 'green', label = 'test')
    
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color = 'blue', label = 'train')
    pyplot.plot(history.history['val_accuracy'], color = 'orange', label = 'test')
    
# run the test harness for evaluating a model
def run_test():
    # define model
    model = define_model()
    
    # create data generator
    datagen = ImageDataGenerator(featurewise_center = True)
    
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.799, 103.939]
    
    # prepare iterator
    train = datagen.flow_from_directory(r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\train-test\train\\",
                                       class_mode = 'categorical', batch_size = 32, target_size = (224, 224))
    test = datagen.flow_from_directory(r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\dataset\train-test\test\\",
                                      class_mode = 'categorical', batch_size = 32, target_size = (224, 224))
    
    # fit model
    history = model.fit(train, steps_per_epoch = len(train), validation_data = test,
                       validation_steps = len(test), epochs = 10, verbose = 1)
    
    # evaluate model and print accuracy
    _, acc = model.evaluate(test, steps = len(test), verbose = 0)
    print('> %.2f' % (acc * 100.0))
    
    # learning curves
    summarise_diagnostics(history)
    
    # save model
    model.save(r"C:\Users\admir\UK\United Kingdom\Personal Projects\Image Classification\image-classification\training_model.h5")
    
# run the test harness
run_test()   
```

You can tweak some of the changes above such as the batch sizes and epochs of the training model if you are not satisfied with the accuracy.
Once the training is completed, you will see the results similar to the screenshots below:

![Training Result](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture15.png)

![Graph](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture16.png)

### User
1. To run the prototype, open the user_interface.ipynb file. You will see the following page:

![user_interface.ipynb](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture17.png)

2. Run the file and a user interface will be displayed.

![User Interface](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture18.png)

3. Search for an image from your device by clicking the 'Upload Image' button. The system will display a File Explorer where you can search and upload an image.

![Search for an image](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture19.png)

4. If you try to upload a file that is not an image (not in the correct image extension, e.g. png, jpg, jpeg), it will display an error message saying "Error!". This will also happen if you try to classify an image ('Classify Image' button) before you upload an image.

![Error Message](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture20.png)

5. When you have chosen an image, the image will be displayed in the user interface.

![Image displayed](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture21.png)

6. Click the 'Classify Image' button and the label classification will be displayed.

![Label classification displayed](https://github.com/Admiral-Ampulembang/image-classification/blob/main/screenshots/Picture22.png)

IMPORTANT NOTE: You will definitely see that there will be cases where the label classification is not correct. You do not have to worry about it because we only uses a low number of images. One way to 'fix' it is to add more images to the dataset (all classes should have a similar number of images so that they are all balanced). You can also experiment with the project by adding more classes (e.g. lion, hyena, etc.) but bear in mind that the more the images in total, the longer it will take to train these images which also depends on the device you are using.

## Credits
- Image Classification (Jason Brownlee) - https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
- Deep CNN Image Classifier (Nicholas Renotte) - https://youtu.be/jztwpsIzEGc?feature=shared"
"# image-classification" 
"# image-classification" 

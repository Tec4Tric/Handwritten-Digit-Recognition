  
'''
This code is written by Sayan De, from Tec4Tric.
Google Colab link - https://bit.ly/mnist-cnn-tec4tric
Website - https://tec4tric.com
Facebook - https://www.facebook.com/tec4tric
YouTube - https://www.youtube.com/tec4tric
Watch this tutorial - 
Accuracy achieved - 99% on training data, 98% on test data
'''


#Importing Packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

#Importing Dataset
from keras.datasets import mnist
(train_img, train_lab), (test_img, test_lab) = mnist.load_data()

#Normalizing Dataset
train_img = train_img.reshape(60000, 28,28,1)
test_img = test_img.reshape(10000,28,28,1)
train_img = keras.utils.normalize(train_img, axis=1)
test_img = keras.utils.normalize(test_img, axis =1)

#Building Model
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(16, (3,3)))
model.add(MaxPooling2D(3,3))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(10, activation="softmax"))

#Compiling Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

#Fitting the Model
model.fit(train_img, train_lab, epochs=10)

#Evaluate the Model
print(model.evaluate(test_img, test_lab))

#Predicting First 10 test images
pred = model.predict(test_img[:10])
# print(pred)
p=np.argmax(pred, axis=1)
print(p)
print(test_lab[:10])

#Visualizing prediction
for i in range(10):
  plt.imshow(test_img[i].reshape((28,28)), cmap='binary')
  plt.title("Original: {}, Predicted: {}".format(test_lab[i], p[i]))
  plt.axis("Off")
  plt.figure()

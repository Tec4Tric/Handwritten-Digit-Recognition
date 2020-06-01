'''
This code is written by Sayan De, from Tec4Tric.
Google Colab link - https://bit.ly/mnist-ann-tec4tric
Website - https://tec4tric.com
Facebook - https://www.facebook.com/tec4tric
YouTube - https://www.youtube.com/tec4tric
Watch this tutorial - https://www.youtube.com/watch?v=FkB0b-jUSbU
Accuracy achieved - 99% on training data, 97% on test data
'''


#Importing Packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

#Importing Dataset
from keras.datasets import mnist
(train_img, train_lab), (test_img, test_lab) = mnist.load_data()

#Normalizing Dataset
train_img = keras.utils.normalize(train_img, axis=1)
test_img = keras.utils.normalize(test_img, axis =1)

#Building Model
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
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
  plt.imshow(test_img[i], cmap='binary')
  plt.title("Original: {}, Predicted: {}".format(test_lab[i], p[i]))
  plt.axis("Off")
  plt.figure()
 

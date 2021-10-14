import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

data = []
labels = []
classes = 43
cur_path = os.getcwd()

print("truy xuất tới ảnh và nhãn")
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
 
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((32,32))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Lỗi load hình ảnh")
 
print("Chuyển đổi danh sách thàng mảng")
data = np.array(data)
print(data.shape)
labels = np.array(labels)
print(labels.shape)
print("Cắt tập train và tập test từ nguồn")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
 
print("Chuyển lable sang dạng One Hot")
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

print("Xây dựng Model")
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
 
print("Tổng hợp Model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Training Model")
epochs = 10
history = model.fit(X_train, y_train, batch_size=10, epochs=epochs, validation_data=(X_test, y_test), shuffle=True)
model.save("my_model.h5")

print("plotting graphs for accuracy ")
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
 
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((32, 32))
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)

# Accuracy with the test data
from sklearn.metrics import accuracy_score

print(accuracy_score(labels, pred))

import pandas as pd
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)

# Accuracy with the test data

print(accuracy_score(labels, pred))
model.save("traffic_classifier.h5")



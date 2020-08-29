# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


# %%
# Load and flatten
data_path = Path(("/home/jovyan/work/CNN Test with Raman/Dataset"
                  "/cells-raman-spectra/dataset_i"))
datasets = data_path.rglob("*.csv")

x_data = [] # Input data, 100cm^-1, 101cm^-1, ..., 2080cm^-1
y_data = [] # Output label,

for item in datasets:
    data = np.loadtxt(item, comments='#', delimiter=',') # size (54, 2090)
    x_data.append(data)
    
    for row in data:
        y_data.append(item.parent.stem.split("-")[0])
x_data = np.concatenate(x_data)

x_data_raw = np.copy(x_data)
y_data_raw = np.copy(y_data)


# %%
print(x_data_raw.shape)
print(y_data_raw)


# %%
# Categories to numbers
le = LabelEncoder()
le.fit(y_data)

classes = le.classes_
print(le.classes_)
num_classes = len(classes)

y_data = le.transform(y_data_raw)

print(le.transform(["A", "DMEM", "G", "HF", "MEL", "ZAM"]))


# %%
# Numbers to one-hot
y_data = to_categorical(y_data, num_classes)
print(y_data[:5])


# %%
# Split into test and training data
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=.2)


# %%
# Check it's worked
print("First 5 traning labels as one-hot encoded vectors:\n",
     y_train[:5])
# Decode
print(le.inverse_transform([np.argmax(train) for train in y_train[:5]]))


# %%
input_shape = 2090

model = Sequential()
model.add(Dense(units=256, activation='sigmoid', input_shape=(input_shape,)))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()


# %%
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy']) # Remember this depends on the loss and optimizer
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=50,
#                     verbose=False,
                    validation_split=0.1) # 10% Validation
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)


# %%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')


# %%
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# input_shape = 28*28
# x_train = x_train.reshape(x_train.shape[0], input_shape)
# x_test = x_test.reshape(x_test.shape[0], input_shape)

predictions = model.predict(x_test)
print(predictions)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

page = 0


# %%
le.inverse_transform([np.argmax(predictions[0])])[0]


# %%
font_dict = {'color': 'black'}
for i in range(16): # Only do first 16
#     ax = plt.subplot(4, 4, i+1)
#     ax.axis('Off')
    plt.plot(np.linspace(100, 4278, 2090), x_test[i + page])
    prediction = le.inverse_transform([np.argmax(predictions[i + page])])[0]
    true_value = le.inverse_transform([np.argmax(y_test[i + page])])[0]
    font_dict['color'] = "black" if prediction == true_value else 'red'
    plt.title(
        f"Predicted: {prediction}, Truth: {true_value}", fontdict=font_dict)
    plt.tight_layout()
    plt.show()
page += 16


# %%




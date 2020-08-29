# -*- coding: utf-8 -*-
# %%
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


# %%
# Load and flatten
data_path = Path("./Dataset/cells-raman-spectra/dataset_i")
datasets = data_path.rglob("*.csv")

x_data = []  # Input data, 100cm⁻¹, 101cm⁻¹, ..., 2080cm⁻¹
y_data = []  # Output label,

for item in datasets:
    data = np.loadtxt(item, comments='#', delimiter=',')  # size (54, 2090)
    x_data.append(data)

    for row in data:
        y_data.append(item.parent.stem.split("-")[0])
x_data = np.concatenate(x_data)

x_data_raw = np.copy(x_data)
y_data_raw = np.copy(y_data)

print(x_data_raw.shape)
print(y_data_raw)


# %%
# Categories to numbers (NOT one encodiing as CrossEntropyLoss doesn't work with it)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_data)

classes = le.classes_
print(le.classes_)
num_classes = len(classes)

y_data = le.transform(y_data_raw)

print(le.transform(list(set(y_data))))


# %%
# Randomly sort, then split into test and training data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    #y_data_ones,
                                                    test_size=.2)
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# %%
plt.plot(x_train[0])


# %%
# Check it's worked
print("First 5 traning labels as one-hot encoded vectors:\n",
     y_train[:5])
# Decode
print(le.inverse_transform(y_train[:5]))


# %%
input_shape = 2090


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x)
        y_pred = self.linear2(h_relu)
        return y_pred

model = TwoLayerNet(2090, 256, num_classes)


# %%
loss_func = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)


# %%
# Train model
for t in range(200):
    # Forwards pass
    y_pred = model(x_train.float())
    
    # Calculate loss
    loss = loss_func(y_pred, y_train)
    
    optimiser.zero_grad()
    
    # Backward pass
    loss.backward()
    
    optimiser.step()

# %%
# Save model
torch.save(model, 'model.pth')

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
for i in range(16):  # Only do first 16
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

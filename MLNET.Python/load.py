from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

basePath = os.getcwd() + '/'
fileSavePath = basePath + 'units/'

file = ''
def getFilePath(unitNumer):
    if not filename.startswith('.'):
        file = "{}/train_unit_{}.txt".format(fileSavePath, str(unitNumber))
    return file

# bepaald bestands lengte
def fileLenght(fname):
    if not fname.startswith('.'):
        print(fname)
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
    return i + 1

# koppel voor elke unit de lengte van het aantal values in een bestand
unitLength = {}
for filename in os.listdir(fileSavePath):
    if not filename.startswith('.'):
        unitNumber = int(os.path.splitext(filename)[0].split('_')[-1])
        unitLength[unitNumber] = fileLenght(os.path.join(fileSavePath, filename))

def create_model():
    model = tf.keras.Sequential([
        keras.layers.LSTM(8, return_sequences=True, input_shape=(None, 24)),
        keras.layers.LSTM(4, input_shape=(None, 24)),
        keras.layers.Dense(1)
        
    ])

    model.compile(optimizer='adam',
                  loss='hinge',
                metrics=['accuracy'])

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
batch_size = 40
unitsFilteredBySequenceLength = []

# verzameld data punten: 
# - lijst van batch size groot
# - - in elke zit een x en y train
# - - - x = input -> sequence lenght * parameters (24)
# - - - y = output -> aantal cycles dat nog over is voor klaar: unitLength - sequence_length
def train_generator():
    while True:
        sequence_length = np.random.randint(10, 300)
        # Defineer de train modules
        x_train = []
        y_train = []
        # get units die beschikbaar zijn binnen de sequence length
        unitsFilteredBySequenceLength = [unitNumber for unitNumber, unitLength in unitLength.items() if unitLength >= sequence_length]

        for _ in range(batch_size):
            unitNumber = random.choice(unitsFilteredBySequenceLength)
            fileLines = []
            with open(getFilePath(unitNumber)) as file:
                fileLines = file.read().splitlines()

            dataStrings = fileLines[0:sequence_length]
            data = []
            for dataString in dataStrings:
                data.append(np.array(list(map(lambda dataPoint: float(dataPoint), dataString.split()[2:]))))
            
            x_train.append(data)
            y_train.append(unitLength[unitNumber] - sequence_length)
            # pak 10 random units met een sequence length van minimaal sequence_length

        yield np.array(x_train), np.array(y_train)


model.fit_generator(train_generator(), steps_per_epoch=30,
                    epochs=10, verbose=1)

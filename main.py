import os
from dataset import maketfdataset
from model import create_model
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_dataset, test_dataset = maketfdataset()
# train  & compile model
model = create_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

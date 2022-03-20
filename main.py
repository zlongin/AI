import pandas as pd
import numpy as np

#i have no clue what to do get me out ASAP

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
                         

losses_train = pd.read_csv(
    "C:\\Users\\zacha\\Desktop\\AI\\russia_losses_equipment.csv",
    names=["day", "aircraft", "helicoptor", "tank", "APC",
           "field artillery", "MRL", "military auto", "fuel tank", "drone", "naval ship", "anti-aircraft warefare"],
    skiprows=1)

losses_train.head()

losses_features = losses_train.copy()

losses_labels = losses_features.pop('aircraft')
losses_labels = losses_features.pop('helicoptor')
losses_labels = losses_features.pop('tank')
losses_labels = losses_features.pop('APC')
losses_labels = losses_features.pop('field artillery')
losses_labels = losses_features.pop('MRL')
losses_labels = losses_features.pop('military auto')
losses_labels = losses_features.pop('fuel tank')
losses_labels = losses_features.pop('drone')
losses_labels = losses_features.pop('naval ship')
losses_labels = losses_features.pop('anti-aircraft warefare')

losses_features = np.array(losses_features)

losses_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

print(losses_features)

losses_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

losses_model.fit(losses_features, losses_labels, epochs=10)






  
  


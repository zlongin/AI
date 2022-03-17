import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import pandas as pd
import numpy as np

footballdata = pd.read_csv("C:\Users\w19023479\Downloads\archive\spreadspoke_scores.csv")
footballdata.head()
footballdata_features = footballdata.copy()
footballdata_homelabel = footballdata_features.pop('score_home');
footballdata_awaylabel = footballdata_features.pop('score_away');


inputs = {}

for name, column in footballdata_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else
        dtype = tf.float32
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(footballdata[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(footballdata_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)
  
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

footballdata_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = footballdata_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

footballdata_features_dict = {name: np.array(value) 
                         for name, value in footballdata_features.items()}
                         
features_dict = {name:values[:1] for name, values in footballdata_features_dict.items()}
footballdata_preprocessing(features_dict)

def footballdatamodel(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
        ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)
    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.optimizers.Adam())
    
    return model
footballdatamodel = footballdatamodel = (footballdata_preprocessing, inputs)

footballdatamodel.fit(x = footballdata_features_dict, y =footballdata_homelabel, epochs = 10)




                         




  
  


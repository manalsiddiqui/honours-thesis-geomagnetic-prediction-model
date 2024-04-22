import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import model

def prep_data(file_path):
    data_frame = pd.read_fwf(file_path)
    
    #Drop the following columns that are not being used as input features
    columns_to_drop = [
        'Timestamp', 'Unnamed: 1', 'Source', 'Bt-med', 'Bt-min', 'Bt-max', 
        'Bx-med', 'Bx-min', 'Bx-max', 'By-med', 'By-min', 'By-max', 'Bz-min', 
        'Bz-max', 'Phi-mean', 'Phi-min', 'Phi-max', 'Theta-med', 'Theta-min', 
        'Theta-max', 'Dens-min', 'Dens-max', 'Speed-min', 'Speed-max', 
        'Temp-min', 'Temp-max'
    ]
    data_frame.drop(columns=columns_to_drop, inplace=True)
    labels = data_frame['Bz-med'].apply(lambda x: 3 if x < -10 else 
                                                2 if x < -20 else 
                                                1 if x < -50 else 0).values
    features = MinMaxScaler().fit_transform(data_frame.drop(columns='Bz-med'))
    
    return features, labels

def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()


# Preparing the data
features, labels = prep_data('maindata.txt')

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Building the model
basic_nn = model.basic_nn(X_train.shape[1], 0.01)

#Training the model
trained_model, history = model.train_model(X_train, y_train, 128, 5, basic_nn)

# Plotting model performance
plot_history(history)


# -*- coding: utf-8 -*-
"""
@author: victor
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.callbacks import ReduceLROnPlateau
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
plt.close("all")

### Fixing the seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))

'''
A custom made stopping criterion
'''
class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=100, min_delta=0.01, verbose=1):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch currently being counted without improvement
        self.wait = 0
        # The epoch the training stops at
        self.stopped_epoch = 0
        # Initialize the best metric as super high
        self.best_mape = np.inf

    def on_epoch_end(self, epoch, logs=None):
        mape=logs.get('val_loss')

        if np.less(mape, self.best_mape - self.min_delta):
            self.best_mape = mape
            self.wait = 0
            # Record the best weights if current results is better
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose ==1: print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose ==1: print("Epoch %d: early stopping" % (self.stopped_epoch + 1))
            
            
def Bejrrum_master(TrainData_in, TrainData_out, dimension):

    # Training metaparameters
    MAX_EPOCH = 3000
    PATIENCE_LR = 100 # before reducing learning rate
    PATIENCE_STOP = 200 # before early stopping
    MIN_DELTA = 0.01 # to define meaningful improvement
        
    # Only unsing one fold for the demo 
    kfold = 5
    rs = KFold(n_splits=kfold, shuffle=True, random_state=0)
    for k, (train_index, test_index) in enumerate(rs.split(TrainData_in.T)):
    
        x_train = np.transpose(TrainData_in[:, train_index])
        x_test = np.transpose(TrainData_in[:, test_index])
        y_train = TrainData_out[dimension, train_index]
        y_test = TrainData_out[dimension, test_index]
       
    nfeature_input = x_train.shape[1]

    # Model structure parameters
    K_NUMBER = 47
    K_WIDTH = 30
    K_STRIDE = 57
    
    K_NUMBER2 = 58
    K_WIDTH2 = 46
    K_STRIDE2 = 34
    
    FC3_DIMS = 235
    DROPOUT = 0.38
    
    # Learning rate
    LRBASE = 3.703e-3
        
    # L2 regularizer parameter
    beta = 15e-3
    K_REG = tf.keras.regularizers.l2(beta)
    
    # Layer initialisation
    K_INIT = tf.keras.initializers.he_normal(seed=42)
    
    # Model structure
    model_cnn = keras.Sequential([  keras.layers.Reshape((nfeature_input, 1),input_shape=(nfeature_input,)), \
                                    keras.layers.Conv1D(filters=K_NUMBER, \
                                                        kernel_size=K_WIDTH, \
                                                        strides=K_STRIDE, \
                                                        padding='same', \
                                                        kernel_initializer=K_INIT,\
                                                        kernel_regularizer=K_REG,\
                                                        activation='elu',\
                                                        ),
                                    keras.layers.Conv1D(filters=K_NUMBER2, \
                                                        kernel_size=K_WIDTH2, \
                                                        strides=K_STRIDE2, \
                                                        padding='same', \
                                                        kernel_initializer=K_INIT,\
                                                        kernel_regularizer=K_REG,\
                                                        activation='elu',\
                                                        ), \
                                    keras.layers.Flatten(),
                                    keras.layers.Dropout(DROPOUT),
                                    keras.layers.Dense(FC3_DIMS, \
                                                       kernel_initializer=K_INIT, \
                                                       kernel_regularizer=K_REG, \
                                                       activation='elu'),
                                    keras.layers.Dense(1, kernel_initializer=K_INIT, \
                                                       kernel_regularizer=K_REG,\
                                                       activation='linear'),
                                  ])
    
    
    # Training
    model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LRBASE), loss='mean_absolute_percentage_error', metrics=['mean_absolute_percentage_error', 'mean_squared_error'])  
    
    # Early stopping to avaid overfitting
    early_stop = CustomEarlyStopping(patience=PATIENCE_STOP, min_delta=MIN_DELTA)
    
    # Reduce LR when stagnating
    rdlr = ReduceLROnPlateau(patience=PATIENCE_LR, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=0)
    
    # Saving on a regular basis
    checkpointer= keras.callbacks.ModelCheckpoint(filepath='CNN/Bjerrum_{:d}.keras'.format(dimension), verbose=1, save_best_only=True)
        
    # Train the model
    model_cnn.fit(x_train, y_train, epochs=MAX_EPOCH, \
              validation_data=(x_test, y_test),  \
              callbacks=[checkpointer, early_stop, rdlr], verbose=1)

    
    # Save the best model
    model_cnn.load_weights('CNN/Bjerrum_{:d}.keras'.format(dimension))

# %%
############
# Training #
############
TrainData = np.load("Database/Database_whole_None_totrain_filtered.npy")
n_output = 4 # size of the y vector
dimension = 0 # 0 => Chlorophyll a, 1 => Chlorophyll b, 2 => Total carotenoids, 3 => Total pigments
TrainData_in = TrainData[:-n_output, :]
TrainData_out = TrainData[-n_output:, :]    

# Train the model
Bejrrum_master(TrainData_in, TrainData_out, dimension)

# %%
##############
# Validation #
##############
ValidData = np.load("Database/Database_run1_None_full_filtered.npy")
ValidData_in = ValidData[:-n_output, :]
ValidData_out = ValidData[-n_output:, :]

#########
# Run 1 #
#########

# Experimental time points
time_vec = np.array([0, 16, 22, 25, 40.5, 45, 49, 65.5, 69, 73, 88.5, 93.5, 97, 163, 168.5, 171.5, 187, 192.5, 205, 215])/24

# Loading the model
model = tf.keras.models.load_model('CNN/Bjerrum_{:d}.keras'.format(dimension))

# Predicting
X = ValidData_in
yPred = model.predict(X.T).flatten()
y = ValidData_out[dimension, :]

# Ploting
plt.figure()
plt.plot(time_vec, y, "og", label = "Methanol extraction")
plt.plot(time_vec, yPred, "or", label = "Opal glass")
plt.xlabel("Time (day)")
plt.ylabel("Cell chlorophyll a content (mg/g)")
plt.xlim([0, 10])
plt.ylim([0, 30])
plt.legend()
plt.savefig("Run1_CNN.png", dpi = 600)
plt.show()



import numpy as np
from pylab import *  # for plot
from scipy.special import factorial
from scipy import linalg as LA
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.special import eval_hermite
from scipy.signal import argrelextrema
import matplotlib
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.special import gamma, factorial
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Build model
class Autoencoder(tf.keras.Model):
    '''
    Subclassed keras tf.keras.Model API. The input will be the potential V(x)
    and the output will be the wave function φ_n(x).
    Args:
      input_size (int): Number of x points
    Attributes:
      input_size (int): Number of x points
      fc1 (layer): First  fully cinnected layer with 512 filters and relu activation function
      dropout1 (layer): Dropout layer with dropout parameter of 0.2
      fc2 (layer): Second  fully cinnected layer with 256 filters and relu activation function
      dropout2 (layer): Dropout layer with dropout parameter of 0.2
      fc3 (layer): Third  fully cinnected layer with 256 filters and relu activation function
      dropout3 (layer): Dropout layer with dropout parameter of 0.2
      fc4 (layer): Fourth  fully cinnected layer with 128 filters and relu activation function
      dropout4 (layer): Dropout layer with dropout parameter of 0.2
      out (layer): Output layer predicting φ_n(x) -> REGRESSION
    '''
    def __init__(self,
                 name='autoencoder', input_size=100,
                 n1 = 256, n2= 256, n3=128, n4=128, drop=0.1, #n's are the neurons per layer
                 **kwargs):
        self.input_size = input_size
        super(Autoencoder, self).__init__(name=name, **kwargséé)

        # Fully connected layer.
        self.fc1 = tf.keras.layers.Dense(n1,  activation='relu') 
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout1 = tf.keras.layers.Dropout(rate=drop)

        # Fully connected layer.
        self.fc2 = tf.keras.layers.Dense(n2,  activation='relu')
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout2 = tf.keras.layers.Dropout(rate=drop)

        # Fully connected layer.
        self.fc3 = tf.keras.layers.Dense(n3, activation='relu')
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout3 = tf.keras.layers.Dropout(rate=drop)

        # Fully connected layer.
        self.fc4 = tf.keras.layers.Dense(n4, activation='relu')
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout4 = tf.keras.layers.Dropout(rate=drop)

        # Output layer (fully connected with input_size neurons and linear activation function )
        self.out = tf.keras.layers.Dense(self.input_size, activation ='linear') 

    @tf.function
    def call(self, inputs, is_training=False):
        '''
        Forward pass of the fully connected model

        Args:
          inputs (tensor): X data to pass through the network (V(x))
          is_training (bool): If training, True, otherwise, False
        
        Returns:
          out (tensor): Output tensor containing the values of φ_n(x)
        '''
        x = tf.reshape(inputs, tf.constant([-1, self.input_size]))
        x = self.fc1(x) # x -> (g(1) o a(1))(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        x = self.fc3(x)
        x = self.dropout3(x, training=is_training)
        x = self.fc4(x)
        x = self.dropout4(x, training=is_training)
        out = self.out(x)
        return out
    




# Training

class Training():
    '''
    Performs the training of the autoencoder model using mean absolute error loss

    Args:
    net (Model): Model to train
    learning_rate (float): Learning Rate for Adam optimizer
    training_iters (int): Numer of training iterations
    batch_size (int): Batch size
    display_step (int): Number of iterations to wait to print the current performance of the model
    early_stopping (int): Number of epochs to wait for the validation loss to increase before performing early stopping
    filepath (str): File path to store and recover the model weights
    restore (bool): If true, it looks for existing weights to reestore them

    Attributes: 
    net (Model): Model to train
    learning_rate (float): Learning Rate for Adam optimizer
    training_iters (int): Numer of training iterations
    batch_size (int): Batch size
    display_step (int): Number of iterations to wait to print the current performance of the model
    stopping_step (int): How many epochs we have waited so far without the validation loss decreasing
    early_stopping (int): Number of epochs to wait for the validation loss to increase before performing early stopping
    filepath (str): File path to store and recover the model weights
    restore (bool): If true, it looks for existing weights to reestore them
    loss (function): Loss function to optimize. In this case, mean square error
    optimizer (tf.Optimizer): Adam optimizer for the learning steps
    ckpt (tf.Checkpoint): Checkpoint that stores weights and optimizer state
    manager (tf.CheckpointManager): Controls that not too many checkpoint files are stored 
    '''
    def __init__(self, 
                 net, #fc_model
                 learning_rate, #alpha
                 training_iters, #max number of epochs
                 batch_size,
                 display_step, 
                 early_stopping=50, #if at a certain point, the validation loss is still growing for the next 50 epochs, you stop
                 filepath=None, 
                 restore =True):
        self.net = net
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step
        self.stopping_step=0
        self.loss = tf.keras.losses.MeanSquaredError()
        self.early_stopping = early_stopping
        self.optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate) # WARNING: we are using the legacy Adam optimizer because of speed issues
        self.filepath = filepath
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory = filepath , max_to_keep=3)  
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

    def loss_val(self, x_val, y_val):
        '''
        Computes the validation loss 
        Args:
        x_val(tensor): batch of validation sample
        y_val (tensor): labels for validation
        Returns:
         val_loss(tensor): validation loss
        '''
        pred_val = self.net(x_val, False) #outside training mode, don't do dropout
        val_loss = self.loss(pred_val, y_val) #prediction, target
        return val_loss

    def early_stop(self, epoch, val_loss, stop):
        '''
        Assesses if we have to stop training
        Args:
         epoch (int): current epoch
         val_loss (tensor): current validation loss
         stop (bool): early stop parameter
        Returns:
         stop(bool): True if the models stops training, false if it continues training
         '''
        #Store best validation loss
        if epoch == 0:
            self.best_loss = val_loss
        else:
            if val_loss < self.best_loss:
                self.stopping_step = 0
                self.best_loss = val_loss
            else:
                #If the validation loss does not decrease, we increase the number of stopping steps
                self.stopping_step += 1
        #If such number reaches the maximum, we stop training
        if self.stopping_step == self.early_stopping:
            stop = True
            print('Early stopping was triggered ')
        return stop

    # Optimization process. 
    @tf.function()
    def run_optimization(self,x, y):
        '''
        Performs one step of the learning process. It calculates the loss function and
        appies backpropagation algorithm to update the weights.

        Args:
        x (tensor): Samples of training data used to train the model
        y (tensor): Labels for training data

        Returns:
        -
        '''
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g: #track how gradient evolves
            # Forward pass.
            pred = self.net(x)
            # Compute loss.
            loss = self.loss(pred, y)

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables) #chain rule

        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss
      
    #@tf.function
    def fit(self, X_train,y_train, X_test,y_test, save=True):
        '''
        Main fit function 

        Args:
          X_train (numpy array): Processed training data
          y_train (numpy array): Labels training data
          X_test (numpy array): Processed test data (validation)
          y_test (numpy array): Labels test data
          save (bool): If true, we save the weights at the end of the training
        Returns:
          -
          '''
        # Create train and test datasets
        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)) 
        train_data = train_data.repeat().shuffle(5000).batch(self.batch_size).prefetch(1) #pick mini batches and shuffles


        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_data = test_data.shuffle(buffer_size=1024).batch(self.batch_size) 

        loss_batch = []
        val_loss_batch = []

        stop = False
        epoch = 0
        # Run training for the given number of steps (and while not early stopping).
        while epoch < self.training_iters and stop == False:
            for step, (batch_x_train, batch_y_train) in enumerate(train_data.take(self.training_iters), 1):
                #Apply backpropagation algorithm
                loss = self.run_optimization(batch_x_train, batch_y_train)
                loss_batch.append(loss.numpy())

                for (test_x, test_y) in test_data:
                    #Compute validation loss
                    val_loss = self.loss_val(test_x, test_y)
                    val_loss_batch.append(val_loss.numpy())

                    stop = self.early_stop(epoch, val_loss, stop)
                    epoch += 1

            #Display the result
            if epoch % self.display_step == 0:
                print('Epoch: ', epoch, "Validation loss: ", val_loss.numpy(), "Loss: ", loss.numpy())

        #Save the weights
        if save:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}".format(save_path))  




def BuildAutoencoder(input_size = 200, dropout = 0.15, learning_rate = 0.001):

    inputs = tf.keras.layers.Input(shape=(None, input_size,) )
    
    X =  tf.keras.layers.Dense(units=256, activation='relu')(inputs)
    X =  tf.keras.layers.Dropout(rate = dropout)(X)

    X =  tf.keras.layers.Dense(units=256, activation='relu')(inputs)
    X =  tf.keras.layers.Dropout(rate = dropout)(X)

    X =  tf.keras.layers.Dense(units=128, activation='relu')(inputs)
    X =  tf.keras.layers.Dropout(rate = dropout)(X)

    X =  tf.keras.layers.Dense(units=128, activation='relu')(inputs)
    X =  tf.keras.layers.Dropout(rate = dropout)(X)

    X =  tf.keras.layers.Dense(units=256, activation='relu')(inputs)
    X =  tf.keras.layers.Dropout(rate = dropout)(X)

    X =  tf.keras.layers.Dense(units=256, activation='relu')(inputs)
    X =  tf.keras.layers.Dropout(rate = dropout)(X)

    outputs = tf.keras.layers.Dense(units =input_size, activation='linear')(X)

    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
    model.compile(loss = tf.keras.losses.MeanSquaredError(), 
                  metrics='loss', 
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate))

    return model




    






from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd 

class Training:
    
    def __init__(self):
        self.dropout = 0.5
        self.time_step = 100
        self.batch_size = 64
        self.epochs = 1
        
        
        
    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
        
    def train(self, model, X_train, y_train, X_test, ytest):
        
        checkpoint = ModelCheckpoint('model_weights/model-{epoch:03d}-{mean_squared_error:03f}-{val_mean_squared_error:03f}.h5', 
                                          verbose=1, monitor='val_mean_squared_error',save_best_only=True, mode='auto')
        lr_reduce = ReduceLROnPlateau(monitor= 'val_mean_squared_error', factor=0.1, 
                                           epsilon=0.01, patience=15, verbose=1)
        
        
        r = model.fit(X_train, y_train, validation_data=(X_test, ytest), 
                      epochs = self.epochs, batch_size = self.batch_size, verbose=1,
                      callbacks = [checkpoint])
        
    # def test(self, X_test, ytest):
    #     model = load_model(self.best_model)
    #     model.evaluate(X_test)
        
        
        
        
if __name__ == "__main__":
    obj = Model()
    model = obj.model_creation()
    
    train_obj = Training()
    
    
    # load the pre-processed dataset
    dataset = pd.read_csv('dataset/pre_processed_dataset.csv')
    print(dataset.shape)
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1,1))
    print(dataset.shape)
    
    # splitting dataset into train and test split
    training_size = int(len(dataset)*0.75)
    test_size = len(dataset) - training_size
    train_data, test_data = dataset[0:training_size,:], dataset[training_size:len(dataset), :1]
    time_step = 100
    print(training_size, test_size, dataset.shape)
    
    X_train, y_train = train_obj.create_dataset(train_data, time_step)
    X_test, ytest = train_obj.create_dataset(test_data, time_step)
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # Model training
    print('------------- Model Training has started ----------------')
    train_obj.train(model, X_train, y_train, X_test, ytest)
    print('------------- Model Training has finished ----------------')
    
    
    
        
        
        
        
        
        


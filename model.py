from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class Model:

    def __init__(self):
        self.dropout = 0.5
        self.optimizer = 'adam'
        self.loss = 'mean_squared_error'
        self.metrics = ['mse', 'mae', 'mape']
        self.time_step = 100
        self.batch_size = 64
        self.epochs = 100
        
    
    def model_creation(self):
        # create stacked LSTM model
        model=Sequential()
        model.add(LSTM(200,return_sequences = True, input_shape = (self.time_step, 1)))
        model.add(Dropout(0.5))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)
        
        model.summary()
        
        return model
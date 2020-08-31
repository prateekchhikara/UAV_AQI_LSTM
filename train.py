from tensorflow.keras.models import Sequential, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


class Model:

    def __init__(self):
        self.dropout = 0.5
        self.optimizer = 'adam'
        self.loss = 'mean_squared_error'
        self.metrics = ['mse', 'mae', 'mape']
        self.time_step = 100
        self.batch_size = 64
        self.epochs = 100
        
        self.checkpoint = ModelCheckpoint('model-{epoch:03d}-{mse:03f}-{val_mse:03f}.h5', 
                                          verbose=1, monitor='val_mse',save_best_only=True, mode='auto')
        self.lr_reduce = ReduceLROnPlateau(monitor= 'val_mse', factor=0.1, 
                                           epsilon=0.01, patience=15, verbose=1)
        
        
    def model_creation(self):
        
        
        # create stacked LSTM model
        model=Sequential()
        model.add(LSTM(200,return_sequences = True, input_shape = (self.time_step, 1)))
        model.add(Dropout(0.5))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)
        
        model.summary()
        
    def train(self, X_train, y_train, X_test, ytest):
        r = model.fit(X_train, y_train, validation_data=(X_test, ytest), 
                      epochs = self.epochs, batch_size = self.batch_size, verbose=1, 
                      callbacks=[self.lr_reduce, self.checkpoint])
        
    def test(self, X_test, ytest):
        model = load_model('model-046-0.008206-0.006697.h5')
        model.evaluate(X_test)
        
        
        
        
        
        


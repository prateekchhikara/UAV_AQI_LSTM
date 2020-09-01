from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import *

class Training:
    
    def __init__(self):
        self.dropout = 0.5
        self.time_step = 100
        self.batch_size = 64
        self.epochs = 100
        
        self.checkpoint = ModelCheckpoint('model_weights/model-{epoch:03d}-{mse:03f}-{val_mse:03f}.h5', 
                                          verbose=1, monitor='val_mse',save_best_only=True, mode='auto')
        self.lr_reduce = ReduceLROnPlateau(monitor= 'val_mse', factor=0.1, 
                                           epsilon=0.01, patience=15, verbose=1)
    
        
    def train(self, model, X_train, y_train, X_test, ytest):
        r = model.fit(X_train, y_train, validation_data=(X_test, ytest), 
                      epochs = self.epochs, batch_size = self.batch_size, verbose=1, 
                      callbacks=[self.lr_reduce, self.checkpoint])
        
    def test(self, X_test, ytest):
        model = load_model(self.best_model)
        model.evaluate(X_test)
        
        
        
        
if __name__ == "__main__":
    obj = Model()
    model = obj.model_creation()
    
    train_obj = Training()
    train_obj.train(model, X_train, y_train, X_test, ytest)
    
    
    
        
        
        
        
        
        


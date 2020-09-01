from keras.models import load_model

class Testing:
    
    def __init__(self):
        self.best_model = '/model_weights/model-046-0.008206-0.006697.h5'

    def test(self, X_test, ytest):
        model = load_model(self.best_model)
        model.evaluate(X_test)
        
        
if __name__ == "__main__":
    obj = Testing()
    obj.test(X_test, ytest)

    
    
    
        
        
        
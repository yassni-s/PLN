import joblib

class Model:
    def __init__(self, path: str):
        self.model = self.load_model(path)

    def load_model(self, path: str):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return None

    def tokenize(self, text):
        if self.model:
            return self.model.tokenize(text)
        else:
            return "Model not loaded"

# Inicialización de las instancias de los modelos
model1 = Model("/Users/YESENIA/Desktop/Taller1/backend/models/model1.pkl")
model2 = Model("/Users/YESENIA/Desktop/Taller1/backend/models/model2.pkl")
model3 = Model("/Users/YESENIA/Desktop/Taller1/backend/models/model3.pkl")

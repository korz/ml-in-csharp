import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import model_from_json
from keras import backend as K

def main():
    #TrainModel()
    UseModel()

def TrainModel():
    #Get Training Data
    input, output = GetTrainingData()

    #Create Model Structure
    model = CreateModel()

    #Train model with the data
    TrainModel(model, input, output)

    #Model Persistence
    modelFilename = R'model.json'
    modelWeightsFilename = R'model-weights.h5'
    SaveModel(model, modelFilename, modelWeightsFilename)

def UseModel():
    modelFilename = R'model.json'
    modelWeightsFilename = R'model-weights.h5'

    #Load Model
    model = LoadModel(modelFilename, modelWeightsFilename)

    #Make Prediction
    print("[%i, %i] = %i" % (0,0, Predict(model, 0,0)))
    print("[%i, %i] = %i" % (0,1, Predict(model, 0,1)))
    print("[%i, %i] = %i" % (1,0, Predict(model, 1,0)))
    print("[%i, %i] = %i" % (1,1, Predict(model, 1,1)))
    print(PredictTensor(model, np.array([[0, 1]])))

def GetTrainingData() -> (np.ndarray, np.ndarray):
    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y = np.array([ 0, 1, 1, 0 ])

    return (x, y)

def CreateModel() -> Sequential:
    model = Sequential()
    model.add(Dense(2))
    model.add(Dense(32, activation= "relu"))
    model.add(Dense(64, activation= "relu"))
    model.add(Dense(1, activation= "sigmoid"))

    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

    return model

def TrainModel(model: Sequential, input: np.ndarray, output: np.ndarray, batchSize: int = 2, epochs: int = 1_000, verbose: int = 1):
    model.fit(input, output, batch_size=batchSize, epochs=epochs, verbose=verbose)

def SaveModel(model: Sequential, modelFilename: str, modelWeightsFilename: str):
    modelJson = model.to_json()
    with open(modelFilename, 'w') as json_file:
        json_file.write(modelJson)
    model.save_weights(modelWeightsFilename)

def LoadModel(modelFilename: str, modelWeightsFilename: str) -> Sequential:
    json_file = open(modelFilename, 'r')
    modelJson = json_file.read()
    json_file.close()
    model = model_from_json(modelJson)
    model.load_weights(modelWeightsFilename)

    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

    return model

def Predict(model: Sequential, p: int, q: int) -> int:
    prediction = model.predict(np.array([[p, q]]))
    predictedValue = prediction[0][0]
    roundedPredictedValue = int(round(predictedValue))
    
    return roundedPredictedValue

def PredictTensor(model: Sequential, tensor: np.ndarray) -> int:
    prediction = model.predict(tensor)
    predictedValue = prediction[0][0]
    roundedPredictedValue = int(round(predictedValue))
    
    return roundedPredictedValue

if __name__ == "__main__":
    main()

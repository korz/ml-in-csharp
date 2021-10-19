import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import model_from_json
from keras import backend as K

import keras2onnx

def main():
    #Load Model
    model = LoadModel()

    #Save to ONNX Format
    onnxModelFilename = R'model.onnx'
    serializedOnnxModel = keras2onnx.convert_keras(model).SerializeToString()

    file = open(onnxModelFilename, "wb")
    file.write(serializedOnnxModel)
    file.close()

def LoadModel() -> Sequential:
    modelFilename = R'model.json'
    modelWeightsFilename = R'model-weights.h5'

    json_file = open(modelFilename, 'r')
    modelJson = json_file.read()
    json_file.close()
    model = model_from_json(modelJson)
    model.load_weights(modelWeightsFilename)

    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    main()

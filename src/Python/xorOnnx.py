import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import model_from_json
from keras import backend as K

import keras2onnx
from onnxruntime import InferenceSession

def main():
    #SaveOnnx()

    UseOnnx()

def SaveOnnx():
    #Load Model
    model = LoadModel()

    #Save to ONNX Format
    onnxModelFilename = R'model.onnx'
    serializedOnnxModel = keras2onnx.convert_keras(model).SerializeToString()

    file = open(onnxModelFilename, "wb")
    file.write(serializedOnnxModel)
    file.close()

def UseOnnx():
    #onnxModelFilename = R'model.onnx'
    onnxModelFilename = R'C:\Code\Talks\MachineLearningCSharp\src\C#\Consoles\MachineLearningCSharp.NetCoreConsole\bin\Debug\netcoreapp3.1\models\model.onnx'
    #onnxModelFilename = R'C:\Code\Talks\MachineLearningCSharp\src\C#\Demos\MachineLearningCSharp.XorMlNet\bin\Debug\netcoreapp3.1\model-mlnet.onnx'

    file = open(onnxModelFilename, "rb")
    onnxModelBytes = file.read()
    file.close()

    session = InferenceSession(onnxModelBytes)

    #Make Prediction
    print("[%i, %i] = %i" % (0, 0, Predict(session, 0, 0)))
    print("[%i, %i] = %i" % (0, 1, Predict(session, 0, 1)))
    print("[%i, %i] = %i" % (1, 0, Predict(session, 1, 0)))
    print("[%i, %i] = %i" % (1, 1, Predict(session, 1, 1)))

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

def Predict(session: InferenceSession, p: int, q: int) -> int:
    x = np.array([[[p, q]]], "float32")
    feed = dict([(sessionInput.name, x[n]) for n, sessionInput in enumerate(session.get_inputs())])

    prediction = session.run(None, feed)
    predictedValue = prediction[0].flatten()[0]
    roundedPredictedValue = int(round(predictedValue))

    return roundedPredictedValue
if __name__ == "__main__":
    main()

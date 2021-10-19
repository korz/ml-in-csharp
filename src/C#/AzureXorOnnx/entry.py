from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

from onnxruntime import InferenceSession

def init():
	global session
	
	onnxModelFilename = R'model.onnx'
    
    file = open(onnxModelFilename, "rb")
    onnxModelBytes = file.read()
    file.close()

    session = InferenceSession(onnxModelBytes)

def run(p, q):
    x = np.array([[[p, q]]], "float32")
    feed = dict([(sessionInput.name, x[n]) for n, sessionInput in enumerate(session.get_inputs())])

    prediction = session.run(None, feed)
    predictedValue = prediction[0].flatten()[0]
    roundedPredictedValue = int(round(predictedValue))

    return roundedPredictedValue

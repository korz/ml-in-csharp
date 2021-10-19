using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MachineLearningCSharp.Onnx
{
    public static class Extensions
    {
        public static IReadOnlyCollection<NamedOnnxValue> CreateTensor(this InferenceSession session, float[] vector, int?[] dimensionShape = null)
        {
            //dictionary= { {"Input 1", 1}, {"Input 2", 1}}

            var dimensionVector = dimensionShape == null
                ? new[] { 1, vector.Length }
                : dimensionShape.Select(x => x.Value).ToArray();

            var dimensions = new ReadOnlySpan<int>(dimensionVector);

            var tensorValues = new List<NamedOnnxValue>();

            foreach (var name in session.InputMetadata.Keys)
            {
                tensorValues.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(vector, dimensions)));
            }

            return tensorValues;
        }

        public static int Predict(this InferenceSession session, int p, int q)
        {
            var tensor = session.CreateTensor(new float[] { p, q });

            var predictedValue = session.Run(tensor).SingleOrDefault()?.AsEnumerable<float>()?.FirstOrDefault() ?? 0;

            var roundedPredictedValue = (int) Math.Round(predictedValue);

            return roundedPredictedValue;
        }
    }
}

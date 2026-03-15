using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MachineLearningCSharp.XorOnnx
{
    class Program
    {
        static void Main(string[] args)
        {
            var session = new InferenceSession(Path.Combine("Models", "model.onnx"));

            //Make Prediction
            Console.WriteLine($"[0, 0] = {session.Predict(0, 0)}");
            Console.WriteLine($"[0, 1] = {session.Predict(0, 1)}");
            Console.WriteLine($"[1, 0] = {session.Predict(1, 0)}");
            Console.WriteLine($"[1, 1] = {session.Predict(1, 1)}");
        }
    }

    public static class Extensions
    {
        public static IReadOnlyCollection<NamedOnnxValue> CreateTensor(this InferenceSession session, float[] vector, int?[] dimensionShape = null)
        {
            //Turns [ 0, 1] into a tensor with name = sequential_1_input and value = {{0, 1}}

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

            var roundedPredictedValue = (int)Math.Round(predictedValue);

            return roundedPredictedValue;
        }
    }
}

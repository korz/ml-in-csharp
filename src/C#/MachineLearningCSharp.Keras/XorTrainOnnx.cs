using System;
using System.IO;
using System.Linq;
using System.Text;
using Keras.Layers;
using Keras.Models;
using Numpy;
using Python.Runtime;
using KerasLib = Keras;

namespace MachineLearningCSharp.Keras
{
    public class XorTrainOnnx
    {
        public static void Run(bool useAnaconda = true)
        {
            if (useAnaconda)
            {
                PythonEngine.PythonHome = @"C:\ProgramData\Miniconda3";
            }

            //You have to initialize in order to use ONNX, regardless of virtual environment
            PythonEngine.Initialize();

            var input = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            var output = np.array(new float[] { 0, 1, 1, 0 });

            var model = new Sequential();
            model.Add(new Dense(2));
            model.Add(new Dense(32, activation: "relu"));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(optimizer: "sgd", loss: "binary_crossentropy", metrics: new[] { "accuracy" });
            model.Fit(input, output, batch_size: 2, epochs: 1_000, verbose: 1);

            //Make Prediction
            Console.WriteLine($"[0, 0] = {Predict(model, 0, 0)}");
            Console.WriteLine($"[0, 1] = {Predict(model, 0, 1)}");
            Console.WriteLine($"[1, 0] = {Predict(model, 1, 0)}");
            Console.WriteLine($"[1, 1] = {Predict(model, 1, 1)}");

            Directory.CreateDirectory("Models");
            var modelFilename = Path.Combine("Models", "model.onnx");

            model.SaveOnnx(modelFilename);
        }

        public static byte[] StringToByteArray(string hex)
        {
            return Enumerable.Range(0, hex.Length)
                .Where(x => x % 2 == 0)
                .Select(x => Convert.ToByte(hex.Substring(x, 2), 16))
                .ToArray();
        }

        public static int Predict(Sequential model, int p, int q)
        {
            var prediction = model.Predict(np.array(new float[,] { { p, q } }));
            var predictedValue = prediction.GetData<float>().SingleOrDefault();
            var roundedPredictedValue = (int)Math.Round(predictedValue);

            return roundedPredictedValue;
        }
    }
}

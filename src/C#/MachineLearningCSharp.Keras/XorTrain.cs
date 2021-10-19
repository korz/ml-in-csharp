using System;
using System.Linq;
using Keras.Layers;
using Keras.Models;
using Numpy;
using Python.Runtime;

namespace MachineLearningCSharp.Keras
{
    public class XorTrain
    {
        public static void Run(bool useAnaconda = true)
        {
            if (useAnaconda)
            {
                PythonEngine.PythonHome = @"C:\ProgramData\Miniconda3";
                PythonEngine.Initialize();
            }

            //Get Training Data
            var input = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            var output = np.array(new float[] { 0, 1, 1, 0 });

            //Create Model Structure
            var model = new Sequential();
            model.Add(new Dense(2));
            model.Add(new Dense(32, activation: "relu"));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(optimizer: "sgd", loss: "binary_crossentropy", metrics: new[] { "accuracy" });

            //Train model with the data
            model.Fit(input, output, batch_size: 2, epochs: 1_000, verbose: 1);

            //Make Prediction
            Console.WriteLine($"[0, 0] = {Predict(model, 0, 0)}");
            Console.WriteLine($"[0, 1] = {Predict(model, 0, 1)}");
            Console.WriteLine($"[1, 0] = {Predict(model, 1, 0)}");
            Console.WriteLine($"[1, 1] = {Predict(model, 1, 1)}");
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

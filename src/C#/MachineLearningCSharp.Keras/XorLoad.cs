using System;
using System.IO;
using System.Linq;
using Keras.Models;
using Numpy;
using Python.Runtime;

namespace MachineLearningCSharp.Keras
{
    public class XorLoad
    {
        public static void Run(bool useAnaconda = true)
        {
            //if (useAnaconda)
            //{
            //    PythonEngine.PythonHome = @"C:\ProgramData\Miniconda3";
            //    PythonEngine.Initialize();
            //}

            var model = BaseModel.ModelFromJson(File.ReadAllText(Path.Combine("Models", "model.json")));
            model.LoadWeight(Path.Combine("Models", "model-weights.h5"));

            //Make Prediction
            Console.WriteLine($"[0, 0] = {Predict(model, 0, 0)}");
            Console.WriteLine($"[0, 1] = {Predict(model, 0, 1)}");
            Console.WriteLine($"[1, 0] = {Predict(model, 1, 0)}");
            Console.WriteLine($"[1, 1] = {Predict(model, 1, 1)}");
        }

        public static int Predict(BaseModel model, int p, int q)
        {
            var prediction = model.Predict(np.array(new float[,] { { p, q } }));
            var predictedValue = prediction.GetData<float>().SingleOrDefault();
            var roundedPredictedValue = (int)Math.Round(predictedValue);

            return roundedPredictedValue;
        }
    }
}

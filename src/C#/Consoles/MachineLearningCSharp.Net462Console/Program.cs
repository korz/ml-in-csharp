using System;

namespace MachineLearningCSharp.Net462Console
{
    public class Program
    {
        static void Main(string[] args)
        {
            //Keras.XorTrain.Run();
            //Keras.XorLoad.Run();

            //Keras.XorTrainOnnx.Run();
            Onnx.XorLoad.Run();

            //Keras.XorTrainOnnx.Run();


            //var modelFilename = @"C:\Code\Talks\MachineLearningCSharp\src\C#\Demos\MachineLearningCSharp.XorMlNet\bin\Debug\netcoreapp3.1\model-mlnet.onnx";
            //var session = new InferenceSession(modelFilename);
            //var prediction = session.Predict(0, 1);


            Console.WriteLine("Hello World!");
        }
    }
}

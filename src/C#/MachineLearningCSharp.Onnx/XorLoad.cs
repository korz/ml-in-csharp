﻿using System;
using System.IO;
using Microsoft.ML.OnnxRuntime;

namespace MachineLearningCSharp.Onnx
{
    public class XorLoad
    {
        public static void Run()
        {
            var session = new InferenceSession(Path.Combine("Models", "model.onnx"));

            //Make Prediction
            Console.WriteLine($"[0, 0] = {session.Predict(0, 0)}");
            Console.WriteLine($"[0, 1] = {session.Predict(0, 1)}");
            Console.WriteLine($"[1, 0] = {session.Predict(1, 0)}");
            Console.WriteLine($"[1, 1] = {session.Predict(1, 1)}");
        }
    }
}

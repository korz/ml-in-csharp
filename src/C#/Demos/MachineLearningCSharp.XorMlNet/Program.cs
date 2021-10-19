using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace MachineLearningCSharp.XorMlNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var examples = new List<Example>
            {
                new Example {P = 0, Q = 0, MutualExclusion = 0},
                new Example {P = 0, Q = 1, MutualExclusion = 1},
                new Example {P = 1, Q = 0, MutualExclusion = 1},
                new Example {P = 1, Q = 1, MutualExclusion = 0},
            };

            //Step 1. Create an ML Context
            var ctx = new MLContext();

            //Step 2. Read in the input data from a text file for model training
            IDataView trainingData = ctx.Data.LoadFromEnumerable(examples);

            //Step 3. Build your data processing and training pipeline
            var pipeline = ctx.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(Example.MutualExclusion));

            //Concat P and Q as Features
            pipeline.Append(ctx.Transforms.Concatenate("Features", nameof(Example.P), nameof(Example.Q)).AppendCacheCheckpoint(ctx));

            //Add Trainer
            pipeline.Append(ctx.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features"));

            //Step 4. Train your model
            ITransformer trainedModel = pipeline.Fit(trainingData);

            ctx.Model.Save(trainedModel, trainingData.Schema, "model.zip");

            var loadedContext = new MLContext();
            DataViewSchema modelSchema;
            var loadedModel = loadedContext.Model.Load("model.zip", out modelSchema);

            //Step 5. Make predictions using your trained model
            var predictionEngine = loadedContext.Model.CreatePredictionEngine<Example, ExamplePrediction>(loadedModel,true);

            var predictions = examples.Select(x =>
            {
                var transformedExample = loadedModel.Transform(loadedContext.Data.LoadFromEnumerable(new[] { x }));

                var convertedData = loadedContext.Data.CreateEnumerable<Example>(transformedExample, true).FirstOrDefault();

                var prediction = predictionEngine.Predict(convertedData);

                return new KeyValuePair<Example, float>(x, prediction.MutualExclusion);
            }).ToList();

            foreach (var prediction in predictions)
            {
                Console.WriteLine($"[{prediction.Key.P}, {prediction.Key.Q}] = {prediction.Value}");
            }
        }
    }
}

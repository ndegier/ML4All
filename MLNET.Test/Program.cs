using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MLNET.Test
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "training_set.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test_set.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            var model = Train(mlContext, _trainDataPath);
            Evaluate(mlContext, model);
            using (FileStream bla = new FileStream(Path.Combine(Environment.CurrentDirectory, "Data", "inputs.txt"), FileMode.Open))
            using (StreamReader reader = new StreamReader(bla))
            {
                while (true)
                {
                    string input = reader.ReadLine();
                    if (string.IsNullOrEmpty(input))
                        break;
                    TestSinglePrediction(mlContext, model, input);
                }
            }
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<MachineLog>(dataPath, hasHeader: false, separatorChar: ',');

            var pipeline = mlContext.Transforms.Concatenate("Features", "Counts", "Max", "Min", "Avg", "Std")
                .Append(mlContext.Regression.Trainers.LightGbm());


            var model = pipeline.Fit(dataView);

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<MachineLog>(_testDataPath, hasHeader: false, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions);

        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model, string input)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<MachineLog, BrokenPrediction>(model);
            IDataView dataView = mlContext.Data.LoadFromTextFile<MachineLog>(_testDataPath, hasHeader: false, separatorChar: ',');

            var bla = dataView.Preview(10);

            var machineLogSample = GetMachineLog(input);

            var prediction = predictionFunction.Predict(machineLogSample);
            Console.WriteLine(prediction.Broken);
        }

        private static MachineLog GetMachineLog(string text)
        {
            var data = text.Split().ToList();

            return new MachineLog
            {
                Counts = data.GetRange(0, 26).Select(s => float.Parse(s)).ToArray(),
                Max = data.GetRange(26, 26).Select(s => float.Parse(s)).ToArray(),
                Min = data.GetRange(52, 26).Select(s => float.Parse(s)).ToArray(),
                Avg = data.GetRange(78, 26).Select(s => float.Parse(s)).ToArray(),
                Std = data.GetRange(104, 26).Select(s => float.Parse(s)).ToArray(),
                Broken = 0
            };
        }
    }
}

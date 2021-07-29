using System;
using NeuralNet.Autodiff;
using NeuralNet;
using NeuralNet.Functions;
using NeuralNet.Optimizers;
using NeuralNet.Loss;
using System.IO;
using System.Globalization;

using System.Linq;

namespace CirclesClassification

{
    class CirclesClassification
    {
        static void Main(string[] args)
        {
            // Load data
            Console.WriteLine("Loading dataset...");
            Tensor xData = new Tensor(LoadData("../dataset/x_data.csv"),requiresGrad:false);
            Tensor yData = new Tensor(LoadData("../dataset/y_data.csv"),requiresGrad:false);

            // Split data into training and test sets
            double split = 0.1;
            int splitIndex = (int)Math.Round(xData.Shape[0]*(1-split));

            Tensor xTrainData = xData.Slice2DTensor(0,splitIndex);
            Tensor yTrainData = yData.Slice2DTensor(0,splitIndex);

            Tensor xTestData = xData.Slice2DTensor(splitIndex,xData.Shape[0]);
            Tensor yTestData = yData.Slice2DTensor(splitIndex,xData.Shape[0]);

            Console.WriteLine("Done.");
            Console.WriteLine(xTestData.Shape[0]);

            // Create a model with the sequential class  
            Sequential model = new Sequential(
                new LinearLayer(2, 4),
                new LeakyRelu(),
                new LinearLayer(4, 8),
                new LeakyRelu(),
                new LinearLayer(8, 2),
                new Sigmoid()

            );

            // Compile the model with the loss function and the optimizer
            Optimizer optimizer = new SGD(0.001);
            ILoss mse = new MSE();
            model.Compile(optimizer,mse);

            // load the data in a dataloader that will split it in multiple batches
            int batchSize = 64;
            DataLoader trainData = new DataLoader(xTrainData,yTrainData,batchSize,true);
            DataLoader testData = new DataLoader(xTestData,yTestData,xTestData.Shape[0],false);

            // Train the model
            int nbEpochs = 130;
            model.Train(trainData,nbEpochs,verbose:true);

            // Evalute the model
            model.Evaluate(testData);      

        }

        // Load the data generated from the python file 'generate_dataset.py' with sklearn
        private static NDimArray LoadData(string filename){
            int nbValues = File.ReadLines(filename).Count();
            NDimArray xData = new NDimArray(new int[]{nbValues,2});
            using(StreamReader reader = new StreamReader(filename)){
                
                double x;
                double y;
                int cnt=0;
                while(!reader.EndOfStream){
                    string[] values = reader.ReadLine().Split(',');
   
                    x = double.Parse(values[0],CultureInfo.InvariantCulture);
                    y = double.Parse(values[1],CultureInfo.InvariantCulture);
                    

                    xData[cnt,0] = x;
                    xData[cnt,1] = y; 

                    cnt++;

                }
            }
    

            return xData;
        }

    }
}

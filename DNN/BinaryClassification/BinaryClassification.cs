using System;
using NeuralNet.Autodiff;
using NeuralNet;
using NeuralNet.Functions;
using NeuralNet.Optimizers;
using NeuralNet.Loss;
using System.IO;
using System.Globalization;

using System.Linq;

namespace BinaryClassification
{
    class BinaryClassification
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Loading dataset...");
            Tensor xData = new Tensor(LoadData("../dataset/x_data.csv"),requiresGrad:true);
            Tensor yData = new Tensor(LoadData("../dataset/y_data.csv"),requiresGrad:true);
            Console.WriteLine("Done.");

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
            DataLoader trainData = new DataLoader(xData,yData,batchSize,true);

            // Train the model
            int nbEpochs = 130;
            model.Train(trainData,nbEpochs,verbose:true);

            // Evalute the model
            model.Evaluate(xData, yData);      

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

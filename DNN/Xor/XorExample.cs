using System;
using NeuralNet;
using NeuralNet.Autodiff;
using NeuralNet.Functions;
using NeuralNet.Loss;
using NeuralNet.Optimizers;
namespace Xor
{
    class XorExample
    {
        static void Main(string[] args)
        {
            XorExample2();
        }

        // This example uses a class to create the model
        public static void XorExample1()
        {
            // xor operation, ydata is encoded this way : [1,0] if 0 and [0,1] if 1
            Tensor xData = new Tensor(requiresGrad: true, shape: new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, shape: new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            // The model class is defined bellow
            MyModel model = new MyModel();

            // Compile the model with the loss function and the optimizer
            Optimizer optimizer = new SGD(0.03);
            ILoss mse = new MSE();
            model.Compile(optimizer, mse);

            // load the data in a dataloader that will split it in multiple batches
            int batchSize = 4;
            DataLoader trainData = new DataLoader(xData, yData, batchSize, true);

            // Train the model
            int nbEpochs = 500;
            model.Train(trainData, nbEpochs, verbose: true);

            // Evalute the model
            model.Evaluate(trainData);
        }

        // This example uses Sequential to define the model
        public static void XorExample2()
        {

            // xor operation, ydata is encoded this way : [1,0] if 0 and [0,1] if 1
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            // Create a model with the sequential class  
            Sequential model = new Sequential(
                new LinearLayer(2, 5),
                new Tanh(),
                new LinearLayer(5, 5),
                new Tanh(),
                new LinearLayer(5, 2),
                new LeakyRelu()
            );

            // Compile the model with the loss function and the optimizer
            Optimizer optimizer = new SGD(0.03);
            ILoss mse = new MSE();
            model.Compile(optimizer, mse);

            // load the data in a dataloader that will split it in multiple batches
            int batchSize = 4;
            DataLoader trainData = new DataLoader(xData, yData, batchSize, true);

            // Train the model
            int nbEpochs = 500;
            model.Train(trainData, nbEpochs, verbose: true);

            // Evalute the model
            model.Evaluate(trainData);

        }

    }

    class MyModel : Model
    {

        public LinearLayer Linear1 { get; set; }
        public LinearLayer Linear2 { get; set; }
        public LinearLayer Linear3 { get; set; }

        public Tanh Activation1 { get; set; }
        public Tanh Activation2 { get; set; }

        public MyModel()
        {
            Linear1 = new LinearLayer(2, 5);
            Activation1 = new Tanh();
            Linear2 = new LinearLayer(5, 5);
            Activation2 = new Tanh();
            Linear3 = new LinearLayer(5, 2);

        }

        public override Tensor Predict(Tensor inputs)
        {
            Tensor output;
            output = Linear1.Forward(inputs);
            output = Activation1.Forward(output);
            output = Linear2.Forward(output);
            output = Activation2.Forward(output);
            output = Linear3.Forward(output);
            return output;
        }


    }
}

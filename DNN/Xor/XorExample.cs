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
            //xor operation, ydata is encoded this way : [1,0] if 0 and [0,1] if 1
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            // The model class is defined bellow
            Model model = new Model();

            Optimizer optimizer = new SGD(0.03);
            ILoss MSE = new MSE();
            int nbEpochs = 1000;
            int batchSize = 1;

            Console.WriteLine(model);

            ModelTrainer modelTrainer = new ModelTrainer(model, xData, yData, nbEpochs, batchSize, optimizer, MSE, shuffle: true, verbose: true);
            modelTrainer.Train();
            modelTrainer.Evaluate(xData, yData);
        }

        // This example uses Sequential to define the model
        public static void XorExample2()
        {
            //xor operation, ydata is encoded this way : [1,0] if 0 and [0,1] if 1
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            Sequential model = new Sequential(
                new LinearLayer(2, 5),
                new Tanh(),
                new LinearLayer(5, 5),
                new Tanh(),
                new LinearLayer(5, 2),
                new LeakyRelu()
            );

            Optimizer optimizer = new SGD(0.03);
            int nbEpochs = 500;
            int batchSize = 1;
            ILoss MSE = new MSE();

            Console.WriteLine(model);

            ModelTrainer modelTrainer = new ModelTrainer(model, xData, yData, nbEpochs, batchSize, optimizer, MSE, shuffle: true, verbose: true);
            modelTrainer.Train();

            DateTime T = DateTime.Now;
     
            modelTrainer.Evaluate(xData, yData);
            
            Console.WriteLine(T - DateTime.Now);

        }


    }

    class Model : Module
    {

        public LinearLayer Linear1 { get; set; }
        public LinearLayer Linear2 { get; set; }
        public LinearLayer Linear3 { get; set; }

        public Tanh Activation1 { get; set; }
        public Tanh Activation2 { get; set; }

        public Model()
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

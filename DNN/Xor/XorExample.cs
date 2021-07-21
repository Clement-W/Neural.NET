using System;
using NeuralNet;
using NeuralNet.Autodiff;
namespace Xor
{
    class XorExample
    {
        static void Main(string[] args)
        {
            //xor operation, ydata is encoded this way : [1,0] if 0 and [0,1] if 1
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor YData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            // The model class is defined bellow
            Model model = new Model();

            Optimizer optimizer = new SGD(0.03);
            int nbEpochs = 1000;

            Console.WriteLine(model);

            //TODO: implement slice for tensor and ndarray to do batching
            
            DateTime t = DateTime.Now;
            for (int epoch = 0; epoch < nbEpochs; epoch++)
            {
                // Set the parameter's gradient to 0
                model.ZeroGrad();

                // Predict with input data
                Tensor predicted = model.Predict(xData);

                // Compute the loss with MSE (compare predicted versus actual)
                Tensor loss = Function.MSE(predicted,YData);

                // Backpropagate the error through gradients
                loss.Backward();

                // Update network parameters with the SGD optimizer
                optimizer.Step(model);

                Console.WriteLine($"Epoch {epoch} : loss = {loss.Data.DataArray[0]}.");
            }
            Console.WriteLine((DateTime.Now-t).Milliseconds);



        }
    }

    class Model : Module
    {

        
        public LinearLayer Linear1 { get; set; }
        public LinearLayer Linear2 { get; set; }
        public LinearLayer Linear3 { get; set; }

        public Model()
        {
            Linear1 = new LinearLayer(2, 5);
            Linear2 = new LinearLayer(5, 5);
            Linear3 = new LinearLayer(5, 2);

        }

        public Tensor Predict(Tensor inputs)
        {
            Tensor output;
            output = Linear1.Forward(inputs);
            output = Function.Tanh(output);
            output = Linear2.Forward(output);
            output = Function.Tanh(output);
            output = Linear3.Forward(output);
            return output;
        }


    }
}

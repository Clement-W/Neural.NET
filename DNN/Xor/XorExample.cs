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
            //xor operation, ydata is encoded this way : [1,0] if 0 and [0,1] if 1
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor YData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            // The model class is defined bellow
            Model model = new Model();

            Optimizer optimizer = new SGD(0.03);
            int nbEpochs = 1000;
            int batch_size = 3;

            Console.WriteLine(model);

            //TODO: implement slice for tensor and ndarray to do batching
            NDimArray startsIndexes = NDimArray.Arange(0,xData.Shape[0],batch_size);

            
            DateTime t = DateTime.Now;
            double epochLoss;
            int endIndex;
            Tensor inputs;
            Tensor predicted;
            Tensor actual;
            Tensor loss;
            ILoss MSE = new MSE();
            for (int epoch = 0; epoch < nbEpochs; epoch++)
            {
                epochLoss = 0;
                startsIndexes.Shuffle();
   
                    
                foreach(int startIndex in startsIndexes.DataArray){
     
                    endIndex = startIndex + batch_size;      

                    // Set the parameter's gradient to 0
                    model.ZeroGrad();

                    // Get inputs for this batch
                    inputs = xData.Slice2DTensor(startIndex,endIndex);
                    // Get actual outputs for this batch
                    actual = YData.Slice2DTensor(startIndex,endIndex);

                    // Predict with input data
                    predicted = model.Predict(inputs);                    

                    // Compute the loss with MSE (compare predicted versus actual)
                    loss = MSE.ComputeLoss(predicted,actual);

                    // Backpropagate the error through gradients
                    loss.Backward();
                    epochLoss+=loss.Data.DataArray[0];

                    // Update network parameters with the SGD optimizer
                    optimizer.Step(model);
                }

                Console.WriteLine($"Epoch {epoch} : loss = {epochLoss}.");
            }
            Console.WriteLine("Training time : " + (DateTime.Now-t).Milliseconds);



        }
    }

    class Model : Module
    {

        
        public LinearLayer Linear1 { get; set; }
        public LinearLayer Linear2 { get; set; }
        public LinearLayer Linear3 { get; set; }

        public Tanh Activation1{get;set;}
        public Tanh Activation2{get;set;}

        public Model()
        {
            Linear1 = new LinearLayer(2, 5);
            Activation1 = new Tanh();
            Linear2 = new LinearLayer(5, 5);
            Activation2 = new Tanh();
            Linear3 = new LinearLayer(5, 2);


        }

        public Tensor Predict(Tensor inputs)
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

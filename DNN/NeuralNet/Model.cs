using NeuralNet.Autodiff;
using NeuralNet.Optimizers;
using NeuralNet.Loss;
using System;
namespace NeuralNet
{
    public abstract class Model : Module
    {
        public Optimizer Optimizer { get; private set; } = null;
        public ILoss LossFunction { get; private set; } = null;

        public abstract Tensor Predict(Tensor inputs);

        public void Compile(Optimizer optimizer, ILoss loss)
        {
            Optimizer = optimizer;
            LossFunction = loss;
        }

        public void Train(DataLoader trainData, int nbEpochs, bool verbose)
        {
            if (Optimizer == null || LossFunction == null)
            {
                throw new InvalidOperationException("You need to compile the model before trianing it.");
            }

            double epochLoss;
            double accuracy=0;
            Tensor inputs;
            Tensor predicted;
            Tensor actual;
            Tensor loss;

            DateTime startTraining = DateTime.Now;
            DateTime startEpoch;
            for (int epoch = 0; epoch < nbEpochs; epoch++)
            {
                startEpoch = DateTime.Now;
                epochLoss = 0;
                foreach (Tuple<Tensor, Tensor> batch in trainData)
                {
                    // Get inputs for this batch
                    inputs = batch.Item1;
                    // Get actual outputs for this batch
                    actual = batch.Item2;

                    // Set the parameter's gradient to 0
                    this.ZeroGrad();

                    // Predict with input data
                    predicted = this.Predict(inputs);

                    // Compute accuracy for this batch
                    accuracy =  Test(predicted,inputs,actual);

                    // Compute the loss with the given loss function (compare predicted versus actual)
                    loss = LossFunction.ComputeLoss(predicted, actual);

                    // Backpropagate the error through gradients
                    loss.Backward();

                    epochLoss += loss.Data.DataArray[0];

                    // Update network parameters with the SGD optimizer
                    Optimizer.Step(this);
                }

                if (verbose)
                {
                    Console.WriteLine($"Epoch {epoch} : accuracy = {accuracy}%, loss = {epochLoss}, time: {(DateTime.Now - startEpoch).TotalMilliseconds}ms");
                }

            }
            if (verbose)
            {
                Console.WriteLine("Training time : " + (DateTime.Now - startTraining).TotalSeconds + " s");
            }


        }


        private double Test(Tensor predictions, Tensor XData, Tensor YData)
        {
            int[] predictedIndexes = predictions.GetPredictionsIndexes();
            int[] ActualIndexes = YData.GetPredictionsIndexes();

            double cntCorrect = 0;
            for (int i = 0; i < YData.Shape[0]; i++)
            {
                if (predictedIndexes[i] == ActualIndexes[i])
                {
                    cntCorrect++;
                }
            }  

            double accuracy = (cntCorrect / (double)XData.Shape[0]) * 100.0;
            return accuracy;
        }

        public void Evaluate(Tensor XTestData, Tensor YTestData)
        {
            Console.WriteLine("\nEvaluation : ");
            Tensor predictions = this.Predict(XTestData);
            double accuracy = Test(predictions, XTestData, YTestData);
            Console.WriteLine("Accuracy : " + accuracy + "%, Loss : " + LossFunction.ComputeLoss(predictions, YTestData).Data.DataArray[0]);

        }


    }
}
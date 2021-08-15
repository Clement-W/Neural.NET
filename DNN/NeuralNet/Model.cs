using NeuralNet.Autodiff;
using NeuralNet.Optimizers;
using NeuralNet.Loss;
using System;
namespace NeuralNet
{
    /// <summary>
    /// This class represents a model, a neural network that can be trained
    /// </summary>
    public abstract class Model : Module
    {
        /// <summary>
        /// The optimizer used to train the network
        /// </summary>
        /// <value></value>
        public Optimizer Optimizer { get; private set; } = null;

        /// <summary>
        /// THe loss function used to train the network
        /// </summary>
        /// <value></value>
        public ILoss LossFunction { get; private set; } = null;

        /// <summary>
        /// Predict the output of the neural network by feeding it with the inputs tensor
        /// </summary>
        /// <param name="inputs">Input tensor</param>
        /// <returns>Output of the neural network</returns>
        public abstract Tensor Predict(Tensor inputs);

        /// <summary>
        /// Compile the model with the opotimizer and the loss.
        /// </summary>
        /// <param name="optimizer"></param>
        /// <param name="loss"></param>
        public void Compile(Optimizer optimizer, ILoss loss)
        {
            Optimizer = optimizer;
            LossFunction = loss;
        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        /// <param name="trainData">Traning set of the dataset</param>
        /// <param name="nbEpochs">Number of epochs</param>
        /// <param name="verbose">Boolean to print informations during the training phase</param>
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
                accuracy = 0;
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
                    accuracy +=  Test(predicted,inputs,actual);

                    // Compute the loss with the given loss function (compare predicted versus actual)
                    loss = LossFunction.ComputeLoss(predicted, actual);

                    // Backpropagate the error through gradients
                    loss.Backward();

                    epochLoss += loss.Data.DataArray[0];

                    // Update network parameters with the SGD optimizer
                    Optimizer.Step(this);
                }

                accuracy = accuracy/trainData.nbBatches;

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


        /// <summary>
        /// Test the neural network
        /// </summary>
        /// <param name="predictions">Predicted values by the neural network</param>
        /// <param name="XData">Input actual data</param>
        /// <param name="YData">Output actual data</param>
        /// <returns></returns>
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

        /// <summary>
        /// Evaluate the neural network on test dataset
        /// </summary>
        /// <param name="testData">The test set of the data</param>
        public void Evaluate(DataLoader testData)
        {
            Console.WriteLine("\nEvaluation : ");
            Tensor predictions = this.Predict(testData.XData);
            double accuracy = Test(predictions, testData.XData, testData.YData);
            Console.WriteLine("Accuracy : " + accuracy + "%, Loss : " + LossFunction.ComputeLoss(predictions, testData.YData).Data.DataArray[0]);

        }


    }
}
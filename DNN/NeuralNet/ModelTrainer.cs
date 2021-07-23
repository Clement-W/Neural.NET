using NeuralNet.Optimizers;
using NeuralNet.Autodiff;
using NeuralNet.Loss;
using System;
namespace NeuralNet
{
    public class ModelTrainer
    {
        public Module Model { get; set; }
        public Optimizer Optimizer { get; set; }
        public ILoss LossFunction { get; set; }
        public int NbEpochs { get; set; }
        public int BatchSize { get; set; }
        public Tensor XData { get; set; }
        public Tensor YData { get; set; }

        public NDimArray startIndexes { get; }

        public bool ShuffleIndexes { get; }

        public bool Verbose { get; set; }


        /**
         * model : The model to train
         * xData : The input data
         * yData : The labels
         * nbEpochs : The number of epochs to train the model
         * batchSize : The batch size
         * Optimizer : The optimizer used to train the network
         * LossFunction : the function used to compute the loss
         * shuffle : If true, shuffle the start indexes of every batch 
         */
        public ModelTrainer(Module model, Tensor xData, Tensor yData, int nbEpochs, int batchSize, Optimizer optimizer, ILoss lossFunction, bool shuffle = true, bool verbose = true)
        {
            Model = model;
            XData = xData;
            YData = yData;
            NbEpochs = nbEpochs;
            BatchSize = batchSize;
            Optimizer = optimizer;
            LossFunction = lossFunction;
            startIndexes = NDimArray.Arange(0, XData.Shape[0], batchSize);
            ShuffleIndexes = shuffle;
            Verbose = verbose;
        }

        //TODO: save trained data (loss, accuracy,...) and plot it with xplot
        public void Train()
        {
            double epochLoss;
            int endIndex;
            Tensor inputs;
            Tensor predicted;
            Tensor actual;
            Tensor loss;


            DateTime startTraining = DateTime.Now;
            for (int epoch = 0; epoch < NbEpochs; epoch++)
            {
                DateTime startEpoch = DateTime.Now;
                epochLoss = 0;
                // Shuffle the start indexes at each epoch

                if (ShuffleIndexes)
                {
                    startIndexes.Shuffle();
                }

                
                foreach (int startIndex in startIndexes.DataArray)
                {
                    endIndex = startIndex + BatchSize;

                    // Set the parameter's gradient to 0
                    Model.ZeroGrad();

                    // Get inputs for this batch
                    inputs = XData.Slice2DTensor(startIndex, endIndex);
                    // Get actual outputs for this batch
                    actual = YData.Slice2DTensor(startIndex, endIndex);

                    // Predict with input data
                    predicted = Model.Predict(inputs);

                    // Compute the loss with MSE (compare predicted versus actual)
                    loss = LossFunction.ComputeLoss(predicted, actual);

                    // Backpropagate the error through gradients
                    loss.Backward();

                    epochLoss += loss.Data.DataArray[0];

                    // Update network parameters with the SGD optimizer

                    Optimizer.Step(Model);
                }

                //TODO: evalueate the model at the end of the epoch to show the accuracy 
                // (the dataloader needs to be implemented before to store traindata and testdata )
                if (Verbose)
                {
                    Console.WriteLine($"Epoch {epoch} : loss = {epochLoss}, time: {(DateTime.Now - startEpoch).TotalMilliseconds}ms");
                }

            }
            if (Verbose)
            {
                Console.WriteLine("Training time : " + (DateTime.Now - startTraining).TotalSeconds + " s");
            }
        }

        public double TestModel(Tensor predictions,Tensor XTestData, Tensor YTestData)
        {
            int[] predictedIndexes = predictions.Data.GetIndexesOfMaxValuesInRowsOf2DArray();
            int[] ActualIndexes = YTestData.Data.GetIndexesOfMaxValuesInRowsOf2DArray();

            double cntCorrect = 0;
            for (int i = 0; i < YTestData.Shape[0]; i++)
            {
                if (predictedIndexes[i] == ActualIndexes[i])
                {
                    cntCorrect++;
                }
            }

            double accuracy = (cntCorrect / (double)XTestData.Shape[0]) * 100.0;
            return accuracy;
        }

        public void Evaluate(Tensor XTestData, Tensor YTestData)
        {
            Console.WriteLine("\nEvaluation : ");
            Tensor predictions = Model.Predict(XTestData);
            double accuracy = TestModel(predictions,XTestData,YTestData);
            Console.WriteLine("Accuracy : " + accuracy + "%, Loss : " + LossFunction.ComputeLoss(predictions, YTestData).Data.DataArray[0]);
       
        }

    }
}
using NeuralNet.Autodiff;
using System.Collections.Generic;
using System;

namespace NeuralNet
{

    /// <summary>
    /// Iterable object to iterate over a dataset composed of x data and y data
    /// </summary>
    public class DataLoader : IEnumerable<Tuple<Tensor, Tensor>>
    {
        /// <summary>
        /// Input data
        /// </summary>
        /// <value></value>
        public Tensor XData { get; }

        /// <summary>
        /// Output data
        /// </summary>
        /// <value></value>
        public Tensor YData { get; }

        /// <summary>
        /// The number of batches that compose the dataset
        /// </summary>
        /// <value></value>
        public int nbBatches { get; }

        /// <summary>
        /// The size of the batches
        /// </summary>
        /// <value></value>
        public int BatchSize { get; }

        /// <summary>
        /// The indexes that delimits batches
        /// </summary>
        /// <value></value>
        public NDimArray StartIndexes { get; }

        /// <summary>
        /// Boolean to specify if the batches will be shuffled
        /// </summary>
        /// <value></value>
        public bool ShuffleIndexes { get; }

        /// <summary>
        /// Constructor used to create a dataloader object 
        /// </summary>
        /// <param name="xData">Input data</param>
        /// <param name="yData">Output data</param>
        /// <param name="batchSize">The size of the batches</param>
        /// <param name="shuffle">Boolean to specify if the batches will be shuffled</param>
        public DataLoader(Tensor xData, Tensor yData, int batchSize, bool shuffle)
        {
            if (xData.Shape[0] != yData.Shape[0])
            {
                throw new ArgumentException("xData and yData should have the same number of samples");
            }
            if (xData.Ndim != 2 && yData.Ndim != 2)
            {
                throw new ArgumentException("The data arrays must be 2 dim arrays.");
            }
            XData = xData;
            YData = yData;
            BatchSize = batchSize;
            ShuffleIndexes = shuffle;
            StartIndexes = NDimArray.Arange(0, XData.Shape[0], BatchSize);
            nbBatches = StartIndexes.NbElements;
        }

        /// <summary>
        /// Returns an enumerator that iterates through the dataloader
        /// The enumerator is a tuple that contains the input and the output data of the edataset
        /// </summary>
        /// <returns>An enumerator</returns>
        public IEnumerator<Tuple<Tensor, Tensor>> GetEnumerator()
        {
            // Shuffle the start indexes before if wanted
            if (ShuffleIndexes)
            {
                StartIndexes.Shuffle();
            }

            int endIndex;

            foreach(int startIndex in StartIndexes.DataArray){
                endIndex  = startIndex + BatchSize;
                yield return new Tuple<Tensor,Tensor>(XData.Slice2DTensor(startIndex,endIndex), YData.Slice2DTensor(startIndex,endIndex));
            }

        }

        /// <summary>
        /// Returns an enumerator that iterates through the dataloader
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }



    }
}
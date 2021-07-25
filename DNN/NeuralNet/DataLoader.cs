using NeuralNet.Autodiff;
using System.Collections.Generic;
using System;

namespace NeuralNet
{
    public class DataLoader : IEnumerable<Tuple<Tensor, Tensor>>
    {
        public Tensor XData { get; }

        public Tensor YData { get; }

        public int nbBatches { get; }

        public int BatchSize { get; }

        public NDimArray StartIndexes { get; }

        public bool ShuffleIndexes { get; }

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

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }



    }
}
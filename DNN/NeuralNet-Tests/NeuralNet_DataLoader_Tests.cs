using Xunit;
using NeuralNet.Autodiff;
using NeuralNet.Functions;
using NeuralNet.Loss;

using System;
using System.Linq;

namespace NeuralNet.UnitTests
{
    public class NeuralNet_DataLoader_Tests
    {

        [Fact]
        public void DataLoader_Test_NbBatches1()
        {
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            int batchSize = 1;
            DataLoader trainData = new DataLoader(xData, yData, batchSize, true);
            Assert.True(trainData.nbBatches==4);

        }

        [Fact]
        public void DataLoader_Test_NbBatches2()
        {
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            int batchSize = 2;
            DataLoader trainData = new DataLoader(xData, yData, batchSize, true);
            Assert.True(trainData.nbBatches==2);

        }

        [Fact]
        public void DataLoader_Test_NbBatches3()
        {
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            int batchSize = 3;
            DataLoader trainData = new DataLoader(xData, yData, batchSize, true);
            Assert.True(trainData.nbBatches==2);

        }

        [Fact]
        public void DataLoader_Test_NbBatches4()
        {
            Tensor xData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
            Tensor yData = new Tensor(requiresGrad: true, new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

            int batchSize = 4;
            DataLoader trainData = new DataLoader(xData, yData, batchSize, true);
            Assert.True(trainData.nbBatches==1);

        }
    }
}

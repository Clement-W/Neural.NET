using Xunit;
using NeuralNet.Autodiff;
using System;
using System.Linq;

namespace NeuralNet.UnitTests
{
    public class NeuralNet_Tensor_Tests
    {

        [Fact]
        public void Tensor_Test_Construcors()
        {
            
            NDimArray a = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray b = new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6);
            Tensor t1 = new Tensor(a);
            Tensor t2 = new Tensor(a, requiresGrad: true);
            TensorDependency[] tdeps = new TensorDependency[] { new TensorDependency(t2, ((w) => w)) };
            Tensor t4 = new Tensor(b, requiresGrad: true,dependencies:tdeps);

            Assert.True(t4.Grad.Shape.SequenceEqual(new int[] { 3, 2 }));
            Assert.True(t4.Grad.Data.Data.SequenceEqual(new double[] { 0,0,0,0,0,0 }));
            Assert.True(t1.Grad == null);

            Console.WriteLine(t4);
        }

        [Fact]
        public void Tensor_Test_Simple_Backward()
        {
            NDimArray b = new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6);
         
            Tensor t4 = new Tensor(b, requiresGrad: true);
            Assert.True(t4.Grad.Data.Data.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0 }));
            t4.Backward(new Tensor(NDimArray.Ones_like(b)));
            Assert.True(t4.Grad.Data.Data.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1 }));

        }

    }
}

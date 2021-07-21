using Xunit;
using NeuralNet.Autodiff;
using System;
using System.Linq;

namespace NeuralNet.UnitTests
{
    public class NeuralNet_LinearLayer_Tests
    {

        [Fact]
        public void LinearLayer_Test_Forward()
        {
            LinearLayer layer = new LinearLayer(1, 1);

            Tensor input = new Tensor(3);
            Tensor output = layer.Forward(input);

            //Console.WriteLine("linear layer output : " + output);

            Assert.True(output.Data.DataArray.SequenceEqual((Tensor.Matmul(input, layer.Weights) + layer.Biases).Data.DataArray));
            Assert.True(output.Shape.SequenceEqual(new int[] { 1 }));


        }

        [Fact]
        public void LinearLayer_Test_Backward()
        {
            LinearLayer layer = new LinearLayer(1, 1);
            layer.Weights.Data = new NDimArray(new int[] { 1, 1 }, 2);

            layer.Biases.Data = new NDimArray(1);
            layer.ZeroGrad();

            Tensor input = new Tensor(3);
            Tensor output = layer.Forward(input);


            Tensor grad = new Tensor(10, 20, 30);
            //Console.WriteLine(layer);
            output.Backward(grad);

            Assert.True(layer.Weights.Grad.Data.DataArray.SequenceEqual(new double[] { 180 }));
            Assert.True(layer.Biases.Grad.Data.DataArray.SequenceEqual(new double[] { 60 }));


        }

        [Fact]
        public void LinearLayer_Test_Parameters()
        {
            LinearLayer layer = new LinearLayer(1, 1);

            Tensor input = new Tensor(3);
            Tensor output = layer.Forward(input);

            Assert.True(layer.Parameters().Count() == 2);

        }

        [Fact]
        public void LinearLayer_Test_Forward_2DLayer()
        {
            LinearLayer layer = new LinearLayer(3, 2);
            layer.Weights.Data = new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6);

            layer.Biases.Data = new NDimArray(1, 1);
            layer.ZeroGrad();


            Tensor input = new Tensor(3, 3, 3);
            Tensor output = layer.Forward(input);

            //Console.WriteLine("linear layer output : " + output);

            Assert.True(output.Data.DataArray.SequenceEqual(new double[] { 28, 37 }));
            Assert.True(output.Shape.SequenceEqual(new int[] { 2 }));

        }


        [Fact]
        public void LinearLayer_Test_Forward_Multiple_Layer()
        {
            LinearLayer layer1 = new LinearLayer(2, 3);
            LinearLayer layer2 = new LinearLayer(3, 5);
            LinearLayer layer3 = new LinearLayer(5, 1);

            Tensor input = new Tensor(3, 3);

            Tensor out1 = layer1.Forward(input);
            Tensor out2 = layer2.Forward(out1);
            Tensor out3 = layer3.Forward(out2);

            //Console.WriteLine(out3);     
        }

    }
}

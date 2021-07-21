using Xunit;
using NeuralNet.Autodiff;
using System;
using System.Linq;

namespace NeuralNet.UnitTests
{
    public class NeuralNet_LinearLayer_Tests
    {
        
        [Fact]
        public void LinearLayer_Test_Forward(){
            LinearLayer layer = new LinearLayer(1,1);

            Tensor input = new Tensor(3);
            Tensor output = layer.Forward(input);

            //Console.WriteLine("linear layer output : " + output);

            //Assert.True(output.Data.DataArray.SequenceEqual((Tensor.Matmul(input,layer.Weights) + layer.Biases).Data.DataArray));
            //Assert.True(output.Shape.SequenceEqual(new int[]{1}));

            
        }

        [Fact]
        public void LinearLayer_Test_Backward(){
            LinearLayer layer = new LinearLayer(1,1);
            layer.Weights.Data = new NDimArray(new int[]{1,1},2);
     
            layer.Biases.Data = new NDimArray(1);
            layer.ZeroGrad();

            Tensor input = new Tensor(3);
            Tensor output = layer.Forward(input);
            // output = 3 @ 2 + 1 = 6 +1 = 7
            // outputshape = (1,1) @ (1,1) + (1) = (1,1)
            //Console.WriteLine(output);

            Tensor grad = new Tensor(10,20,30);
            //Console.WriteLine(layer);
            output.Backward(grad);

            //Console.WriteLine("---");
            //Console.WriteLine(layer.Weights.Grad);
            //Console.WriteLine(layer.Biases.Grad);


            //Assert.True(layer.Weights.Grad.Data.DataArray.SequenceEqual(new double[]{3}));
            //Assert.True(layer.Biases.Grad.Data.DataArray.SequenceEqual(new double[]{1}));
            //Assert.True(input.Grad == null);
            
        }

        [Fact]
        public void LinearLayer_Test_Parameters(){
            LinearLayer layer = new LinearLayer(1,1);

            Tensor input = new Tensor(3);
            Tensor output = layer.Forward(input);

            //Assert.True(layer.Parameters().Count()==2);
            
        }

    }
}

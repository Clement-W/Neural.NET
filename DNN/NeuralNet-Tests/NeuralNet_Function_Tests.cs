using Xunit;
using NeuralNet.Autodiff;
using NeuralNet.Functions;
using NeuralNet.Loss;

using System;
using System.Linq;

namespace NeuralNet.UnitTests
{
    public class NeuralNet_Function_Tests
    {

        [Fact]
        public void Function_Test_Tanh()
        {
            Tensor t = new Tensor(requiresGrad:true,1,2,3);
            Tensor tanh = new Tanh().Forward(t);

            Assert.True(tanh.Data.DataArray.SequenceEqual(NDimArray.Tanh(t.Data).DataArray));

            Tensor grad = new Tensor(1,1,1);
            tanh.Backward(grad);

            Assert.True(t.Grad.Data.DataArray.SequenceEqual((grad.Data*(1-tanh.Data*tanh.Data)).DataArray));

        }


        [Fact]
        public void Function_Test_LeakyRelu()
        {
            Tensor t = new Tensor(requiresGrad:true,1,-2,0);
            Tensor lrelu = new LeakyRelu().Forward(t);

            Assert.True(lrelu.Data.DataArray.SequenceEqual(NDimArray.LeakyRelu(t.Data).DataArray));

            Tensor grad = new Tensor(2,2,2);
            lrelu.Backward(grad);
            
            Assert.True(t.Grad.Data.DataArray.SequenceEqual(new double[]{2,0.02,2}));
        }


        [Fact]
        public void Function_Test_Sigmoid()
        {
            Tensor t = new Tensor(requiresGrad:true,1,2,3);
            Tensor sigmoid = new Sigmoid().Forward(t);
            Assert.True(sigmoid.Data.DataArray.SequenceEqual((1/(1+NDimArray.Exp(-t.Data))).DataArray));

            Tensor grad = new Tensor(2,2,2);
            sigmoid.Backward(grad);
            
            Assert.True(t.Grad.Data.DataArray.SequenceEqual((grad.Data * (sigmoid.Data*(1-sigmoid.Data))).DataArray));

        }

        [Fact]
        public void Function_Test_MSE()
        {
            Tensor actual = new Tensor(requiresGrad:true,1,2,3);
            Tensor predicted = new Tensor(requiresGrad:true,1.1,2.1,3.1);
            
            Tensor mse = new MSE().ComputeLoss(predicted,actual);
            Assert.True(mse.Data.DataArray.SequenceEqual(((predicted-actual)*(predicted-actual)).Sum().Data.DataArray));

        }

        [Fact]
        public void Function_Test_BinaryCrossentropy()
        {
            Tensor actual = new Tensor(requiresGrad:true,1,0,1);
            Tensor predicted = new Tensor(requiresGrad:true,0.8,0.1,0.9);
            
            Tensor crossentropy = new BinaryCrossentropy().ComputeLoss(predicted,actual);
            Console.WriteLine(crossentropy);
            Assert.Equal(expected:"0,14462152754328741", actual:crossentropy.Data.DataArray[0].ToString());
        
        }

    }
}

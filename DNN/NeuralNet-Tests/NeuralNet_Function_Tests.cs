using Xunit;
using NeuralNet.Autodiff;

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
            Tensor tanh = Function.Tanh(t);

            Assert.True(tanh.Data.DataArray.SequenceEqual(NDimArray.Tanh(t.Data).DataArray));

            Tensor grad = new Tensor(1,1,1);
            tanh.Backward(grad);

            Assert.True(t.Grad.Data.DataArray.SequenceEqual((grad.Data*(new NDimArray(1)-tanh.Data*tanh.Data)).DataArray));

        }


        [Fact]
        public void Function_Test_LeakyRelu()
        {
            Tensor t = new Tensor(requiresGrad:true,1,-2,0);
            Tensor lrelu = Function.LeakyRelu(t);

            Assert.True(lrelu.Data.DataArray.SequenceEqual(NDimArray.LeakyRelu(t.Data).DataArray));

            Tensor grad = new Tensor(2,2,2);
            lrelu.Backward(grad);
            
            Assert.True(t.Grad.Data.DataArray.SequenceEqual(new double[]{2,0.02,2}));
        }


        [Fact]
        public void Function_Test_Sigmoid()
        {
            Tensor t = new Tensor(requiresGrad:true,1,2,3);
            Tensor sigmoid = Function.Sigmoid(t);
            Assert.True(sigmoid.Data.DataArray.SequenceEqual((new NDimArray(1)/(new NDimArray(1)+NDimArray.Exp(-t.Data))).DataArray));

            Tensor grad = new Tensor(2,2,2);
            sigmoid.Backward(grad);
            
            Assert.True(t.Grad.Data.DataArray.SequenceEqual((grad.Data * (sigmoid.Data*(new NDimArray(1)-sigmoid.Data))).DataArray));

        }

        [Fact]
        public void Function_Test_MSE()
        {
            Tensor actual = new Tensor(requiresGrad:true,1,2,3);
            Tensor predicted = new Tensor(requiresGrad:true,1.1,2.1,3.1);
            
            Tensor mse = Function.MSE(predicted,actual);
            Assert.True(mse.Data.DataArray.SequenceEqual(((predicted-actual)*(predicted-actual)).Sum().Data.DataArray));

        }

    }
}

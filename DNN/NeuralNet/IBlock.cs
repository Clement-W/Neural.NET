using NeuralNet.Autodiff;
using System;
namespace NeuralNet
{
    // Basic building block for neural networks
    public interface IBlock
    {
        Tensor Forward(Tensor inputs);
        
    }
}
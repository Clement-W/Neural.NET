using NeuralNet.Autodiff;
using System;
namespace NeuralNet
{
    /// <summary>
    /// Basic building block for neural networks
    /// </summary>
    public interface IBlock
    {
        /// <summary>
        /// Forward an input tensor into the block, and return the output tensor
        /// </summary>
        /// <param name="inputs">The input tensor</param>
        /// <returns>The output tensor</returns>
        Tensor Forward(Tensor inputs);
        
    }
}
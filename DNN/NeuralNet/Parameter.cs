using NeuralNet.Autodiff;
using System;
namespace NeuralNet
{

    /// <summary>
    /// A parameter is a tensor used in neural networks, so it requires the gradient computation
    /// and is initialized randomly or in a specific way according to the parameter's type
    /// </summary>
    public class Parameter : Tensor
    {
    
        /// <summary>
        /// This constructor create a tensor composed of a randomized ndimarray with the given shape
        /// </summary>
        /// <param name="shape">The shape of the parameter</param>
        public Parameter(params int[] shape)
        : base(NDimArray.Random(shape), requiresGrad: true) { }

        /// <summary>
        /// This constructor create a tensor composed of the given ndimarray
        /// </summary>
        /// <param name="arr">A ndim array that compose the parameter</param>
        public Parameter(NDimArray arr) : base(arr,requiresGrad:true){}

        /// <summary>
        /// Initialize the weights
        /// </summary>
        /// <param name="inputSize"></param>
        /// <param name="outputSize"></param>
        /// <returns></returns>
        public static Parameter InitializeWeights(int inputSize, int outputSize){
            // Initilize weights to a random number in [-y,y]
            // with y = 1/sqrt(number of inputs)
            double y = 1/Math.Sqrt(inputSize);
            NDimArray res = NDimArray.Random(shape:new int[]{inputSize,outputSize},-y,y);
            return new Parameter(res);      
        }

        /// <summary>
        /// Initialize the biases
        /// </summary>
        /// <param name="nbNeurons"></param>
        /// <returns></returns>
        public static Parameter InitializeBiases(int nbNeurons){
            // Initialize biases to 0
            return new Parameter(new NDimArray(new int[]{nbNeurons}));
        }
    }
}
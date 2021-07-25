using NeuralNet.Autodiff;
using System;
namespace NeuralNet
{
    // A parameter is a tensor used in neural networks, so it requires the gradient computation
    // and is randomly initialized
    public class Parameter : Tensor
    {
        // This constructor create a tensor composed of a randomized ndimararay with the given shape
        public Parameter(params int[] shape)
        : base(NDimArray.Random(shape), requiresGrad: true) { }

        public Parameter(NDimArray arr) : base(arr,requiresGrad:true){}

        public static Parameter InitializeWeights(int inputSize, int outputSize){
            // Initilize weights to a random number in [-y,y]
            // with y = 1/sqrt(number of inputs)
            double y = 1/Math.Sqrt(inputSize);
            NDimArray res = NDimArray.Random(shape:new int[]{inputSize,outputSize},-y,y);
            return new Parameter(res);      
        }

        public static Parameter InitializeBiases(int nbNeurons){
            // Initialize biases to 0
            return new Parameter(new NDimArray(new int[]{nbNeurons}));
        }
    }
}
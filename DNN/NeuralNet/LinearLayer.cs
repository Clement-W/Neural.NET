using NeuralNet.Autodiff;
namespace NeuralNet
{
    /// <summary>
    /// Linear layer of a neural net. Apply a linear transformation to the incoming data
    // y = input @ W + b
    /// </summary>
    public class LinearLayer : Module,IBlock
    {
        /// <summary>
        /// The weights of the layer
        /// </summary>
        /// <value></value>
        public Parameter Weights{get;set;}

        /// <summary>
        /// The biases of the layer
        /// </summary>
        /// <value></value>
        public Parameter Biases{get;set;}

        /// <summary>
        /// The input size of the layer
        /// </summary>
        /// <value></value>
        public int InputSize{get;}

        /// <summary>
        /// The output size of the layer
        /// </summary>
        /// <value></value>
        public int OutputSize{get;}

        /// <summary>
        /// Constructor used to create a Linear layer with input and output size
        /// </summary>
        /// <param name="inputSize">The number of inputs of the layer</param>
        /// <param name="outputSize">The number of outputs of the layer</param>
        public LinearLayer(int inputSize,int outputSize){
            InputSize = inputSize;
            OutputSize = outputSize;
            this.Weights = Parameter.InitializeWeights(inputSize,outputSize); // (inputSize,outputSize) matrix
            this.Biases = Parameter.InitializeBiases(outputSize); //outputSize = nb of neurons
        }

        /// <summary>
        /// Forward an input tensor into the linear layer and applies input @ Weights + Biases
        /// </summary>
        /// <param name="inputs">Input tensor</param>
        /// <returns>Returns input @ Weihts + Biases</returns>
        public Tensor Forward(Tensor inputs){
            return Tensor.Matmul(inputs,Weights) + Biases;
        }

    }
}
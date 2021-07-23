using NeuralNet.Autodiff;
namespace NeuralNet
{
    // Linear layer of a neural net, apply a linear transformation to the incoming data
    // y = input @ W + b
    public class LinearLayer : Module,IBlock
    {
        public Parameter Weights{get;set;}

        public Parameter Biases{get;set;}

        public int InputSize{get;}
        public int OutputSize{get;}

        public LinearLayer(int inputSize,int outputSize){
            InputSize = inputSize;
            OutputSize = outputSize;
            this.Weights = new Parameter(inputSize,outputSize); // (inputSize,outputSize) matrix
            this.Biases = new Parameter(outputSize); //outputSize = nb of neurons
        }

        public Tensor Forward(Tensor inputs){
            return Tensor.Matmul(inputs,Weights) + Biases;
        }
    }
}
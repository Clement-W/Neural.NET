using NeuralNet.Autodiff;
namespace NeuralNet
{
    // Linear layer of a neural net, apply a linear transformation to the incoming data
    // y = input @ W + b
    public class LinearLayer : Module,IBlock
    {
        public Parameter Weights{get;set;}

        public Parameter Biases{get;set;}

        public LinearLayer(int input_size,int output_size){
            this.Weights = new Parameter(input_size,output_size); // (input_size,output_size) matrix
            this.Biases = new Parameter(output_size); //output_size = nb of neurons
        }

        public Tensor Forward(Tensor inputs){
            return Tensor.Matmul(inputs,Weights) + Biases;
        }
    }
}
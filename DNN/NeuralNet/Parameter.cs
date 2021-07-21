using NeuralNet.Autodiff;
namespace NeuralNet
{
    // A parameter is a tensor used in neural networks, so it requires the gradient computation
    // and is randomly initialized
    public class Parameter : Tensor
    {
        // This constructor create a tensor composed of a randomized ndimararay with the given shape
        public Parameter(params int[] shape)
        : base(NDimArray.Random(shape), requiresGrad: true) { }
    }
}
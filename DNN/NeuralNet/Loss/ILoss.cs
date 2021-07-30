using NeuralNet.Autodiff;
namespace NeuralNet.Loss
{
    public interface ILoss
    {
        Tensor ComputeLoss(Tensor predicted, Tensor target);
    }
}
using NeuralNet.Autodiff;
namespace NeuralNet.Loss
{
    public class MSE : ILoss
    {
        public Tensor ComputeLoss(Tensor input, Tensor target)
        {
            // We don't need to give a gradient function, we are only using basics operations
            // that has already been defined with a gradient function, so the gradient can be
            // computed automatically
            Tensor errors = input - target;
            Tensor loss = (errors * errors).Sum();
            return loss;
        }
    }
}
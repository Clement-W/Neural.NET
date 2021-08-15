using NeuralNet.Autodiff;
namespace NeuralNet.Loss
{
    /// <summary>
    /// Mean squared error function
    /// </summary>
    public class MSE : ILoss
    {

        /// <summary>
        /// Compute the binary MSE loss thanks to the predicted values and the target values
        /// </summary>
        /// <param name="predicted">The predicted values</param>
        /// <param name="target">The actual values</param>
        /// <returns>The loss</returns>
        public Tensor ComputeLoss(Tensor predicted, Tensor target)
        {
            // We don't need to give a gradient function, we are only using basics operations
            // that has already been defined with a gradient function, so the gradient can be
            // computed automatically
            Tensor errors = predicted - target;
            Tensor loss = (errors * errors).Sum();
            return loss;
        }
    }
}
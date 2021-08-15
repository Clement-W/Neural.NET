using NeuralNet.Autodiff;
namespace NeuralNet.Loss
{
    /// <summary>
    /// Loss function
    /// </summary>
    public interface ILoss
    {
        /// <summary>
        /// Compute the loss thanks to the predicted values and the target values
        /// </summary>
        /// <param name="predicted">The predicted values</param>
        /// <param name="target">The actual values</param>
        /// <returns>The loss</returns>
        Tensor ComputeLoss(Tensor predicted, Tensor target);
    }
}
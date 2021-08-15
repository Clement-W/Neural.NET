using NeuralNet.Autodiff;
using System;
namespace NeuralNet.Loss
{
    /// <summary>
    /// Binary Crossentropy loss function
    /// </summary>
    public class BinaryCrossentropy : ILoss
    {

        /// <summary>
        /// Compute the binary crossentropy loss thanks to the predicted values and the target values
        /// </summary>
        /// <param name="predicted">The predicted values</param>
        /// <param name="target">The actual values</param>
        /// <returns>The loss</returns>
        public Tensor ComputeLoss(Tensor predicted, Tensor target)
        {
            // We don't need to give a gradient function, we are only using basics operations
            // that has already been defined with a gradient function, so the gradient can be
            // computed automatically
            double f = -1.0/predicted.Shape[0];
            
            Tensor loss = f * (target * Tensor.Log(predicted) + (1-target)*Tensor.Log(1-predicted)).Sum();
            return loss;
        }
    }
}
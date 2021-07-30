using NeuralNet.Autodiff;
using System;
namespace NeuralNet.Loss
{
    public class BinaryCrossentropy : ILoss
    {
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
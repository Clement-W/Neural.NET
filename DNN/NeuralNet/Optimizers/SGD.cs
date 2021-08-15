using System;
namespace NeuralNet.Optimizers
{
    /// <summary>
    /// Stochastic gradient descent optimization algorithm
    /// </summary>
    public class SGD : Optimizer
    {
        /// <summary>
        /// Constructor to create a SGD optimizer with a learning rate
        /// </summary>
        /// <param name="lr"></param>
        /// <returns></returns>
        public SGD(double lr) : base(lr){}


        /// <summary>
        /// Perform a parameter update, based on the current gradient of the parameters
        /// </summary>
        /// <param name="module">The module on which the parameters will be updated</param>
        public override void Step(Module module){
            foreach(Parameter param in module.Parameters()){            
                param.Data -= param.Grad.Data * LearningRate;
            }
        }
    }
}
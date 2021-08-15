using NeuralNet.Autodiff;
namespace NeuralNet.Optimizers
{
    /// <summary>
    /// Abstract class used to implement optimization algorithms
    /// </summary>
    public abstract class Optimizer
    {
        /// <summary>
        /// The learning rate
        /// </summary>
        /// <value></value>
        public NDimArray LearningRate{get;set;}

        /// <summary>
        /// Constructor used to create an optimizer object by specifying the learning rate
        /// </summary>
        /// <param name="lr"></param>
        public Optimizer(double lr){
            LearningRate = new NDimArray(lr);
        }

        /// <summary>
        /// Perform a parameter update, based on the current gradient of the parameters
        /// </summary>
        /// <param name="module">The module on which the parameters will be updated</param>
        public abstract void Step(Module module);
    }
}
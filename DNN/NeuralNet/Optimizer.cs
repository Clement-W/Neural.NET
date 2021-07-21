using NeuralNet.Autodiff;
namespace NeuralNet
{
    public abstract class Optimizer
    {
        public NDimArray LearningRate{get;set;}

        public Optimizer(int lr){
            LearningRate = new NDimArray(lr);
        }

        // Perform a paramater update, based on the current gradient of the parameters
        public abstract void Step(Module module);
    }
}
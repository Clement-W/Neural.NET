using System;
namespace NeuralNet.Optimizers
{
    public class SGD : Optimizer
    {
        public SGD(double lr) : base(lr){}

        public override void Step(Module module){
            foreach(Parameter param in module.Parameters()){            
                param.Data -= param.Grad.Data * LearningRate;
            }
        }
    }
}
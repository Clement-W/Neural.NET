namespace NeuralNet
{
    public class SGD : Optimizer
    {
        public SGD(int lr) : base(lr){}

        public override void Step(Module module){
            foreach(Parameter param in module.Parameters()){
                param.Data -= param.Grad.Data * LearningRate;
            }
        }
    }
}
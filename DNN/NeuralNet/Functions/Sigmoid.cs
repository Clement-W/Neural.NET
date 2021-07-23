using NeuralNet.Autodiff;
namespace NeuralNet.Functions
{
    public class Sigmoid : IBlock
    {
        public Tensor Forward(Tensor t){
            
            NDimArray data = 1/(1+NDimArray.Exp(-t.Data));

            TensorDependency[] dependencies = null;

            if (t.RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    // derivataive of sigmoid(x) is sigmoid(x)*(1-sigmoid(x))
                    return incomingGrad * (data*(1-data));
                }

                dependencies = new TensorDependency[] { new TensorDependency(t, GradientFunction) };

            }

            return new Tensor(data, t.RequiresGrad, dependencies);
        }
    }
}
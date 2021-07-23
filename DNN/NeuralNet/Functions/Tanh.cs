using NeuralNet.Autodiff;
namespace NeuralNet.Functions
{
    public class Tanh : IBlock
    {
        public Tensor Forward(Tensor t){
            
            NDimArray data = NDimArray.Tanh(t.Data);
            TensorDependency[] dependencies = null;

            if (t.RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    // derivataive of tanh(x) is (1-tanh^2(x))
                    return incomingGrad * (1 - data * data);
                }

                dependencies = new TensorDependency[] { new TensorDependency(t, GradientFunction) };

            }

            return new Tensor(data, t.RequiresGrad, dependencies);
        }
    }
}
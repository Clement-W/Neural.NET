using NeuralNet.Autodiff;
namespace NeuralNet.Functions
{
    /// <summary>
    /// Sigmoid activation function
    /// </summary>
    public class Sigmoid : IBlock
    {
        /// <summary>
        /// Forward an input tensor into the Sigmoid function 
        /// (applies the element-wise sigmoid function to the input tensor)
        /// </summary>
        /// <param name="inputs">The input tensor</param>
        /// <returns>The output tensor</returns>
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
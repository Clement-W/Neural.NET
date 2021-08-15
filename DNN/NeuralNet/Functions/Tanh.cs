using NeuralNet.Autodiff;
namespace NeuralNet.Functions
{
    /// <summary>
    /// Tanh activation function
    /// </summary>
    public class Tanh : IBlock
    {
        /// <summary>
        /// Forward an input tensor into the Tanh function 
        /// (applies the element-wise Tanh function to the input tensor)
        /// </summary>
        /// <param name="inputs">The input tensor</param>
        /// <returns>The output tensor</returns>
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
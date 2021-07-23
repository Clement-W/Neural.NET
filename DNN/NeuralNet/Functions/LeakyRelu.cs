using NeuralNet.Autodiff;
namespace NeuralNet.Functions
{
    public class LeakyRelu : IBlock
    {
        public Tensor Forward(Tensor t){
            NDimArray data = NDimArray.LeakyRelu(t.Data);
            TensorDependency[] dependencies = null;

            if (t.RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    // derivataive of leaky_relu(x) is 0.01 if x <0, and 1 if x>=0
                    NDimArray res = new NDimArray(t.Shape);
                    for(int i =0;i<res.NbElements;i++){
                        if(t.Data.DataArray[i]>=0){
                            res.DataArray[i]= 1;
                        }else{
                            res.DataArray[i] = 0.01;
                        }
                    }
                    return incomingGrad * res;
                }

                dependencies = new TensorDependency[] { new TensorDependency(t, GradientFunction) };

            }

            return new Tensor(data, t.RequiresGrad, dependencies);
        
        }
    }
}
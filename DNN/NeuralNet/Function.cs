using NeuralNet.Autodiff;
namespace NeuralNet
{
    public class Function
    {
        public static Tensor Tanh(Tensor t)
        {
            NDimArray data = NDimArray.Tanh(t.Data);
            TensorDependency[] dependencies = null;

            if (t.RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    // derivataive of tanh(x) is (1-tanh^2(x))
                    return incomingGrad * (new NDimArray(1) - data * data);
                }

                dependencies = new TensorDependency[] { new TensorDependency(t, GradientFunction) };

            }

            return new Tensor(data, t.RequiresGrad, dependencies);
        }

        public static Tensor LeakyRelu(Tensor t)
        {
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


        public static Tensor Sigmoid(Tensor t)
        {
            NDimArray data = new NDimArray(1)/(new NDimArray(1)+NDimArray.Exp(-t.Data));

            TensorDependency[] dependencies = null;

            if (t.RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    // derivataive of sigmoid(x) is sigmoid(x)*(1-sigmoid(x))
                    return incomingGrad * (data*(new NDimArray(1)-data));
                }

                dependencies = new TensorDependency[] { new TensorDependency(t, GradientFunction) };

            }

            return new Tensor(data, t.RequiresGrad, dependencies);
        }

        public static Tensor MSE(Tensor predicted,Tensor actual){
            // We don't need to give a gradient function, we are only using basics operations
            // that has already been defined with a gradient function, so the gradient can be
            // computed automatically
            Tensor errors = predicted - actual;
            Tensor loss = (errors*errors).Sum();
            return loss;
        }
    }
}
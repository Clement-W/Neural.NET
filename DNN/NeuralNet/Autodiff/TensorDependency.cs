using System;

namespace NeuralNet.Autodiff
{
    public class TensorDependency
    {
        /**
         * This class is used to keep track of the tensor's dependency. For example, if a tensor is made up by the sum of
         * two other tensor, this tensor will have two TensorDependency object in it's list of dependency. We also need to 
         * save the gradient function of this tensor. If we have the gradient of a loss function iwth respect of their sum,
         * we can use the grad_fn functions to get the gradient of the loss function, with respect of each of the original input tensors.
         */

        // The tensor dependence
        public Tensor TensorDep{get;set;}

        // The gradient function, takes a NDimArray as input, and ouput a NDimArray
        public Func<NDimArray,NDimArray> GradFunction { get; set; }

        public TensorDependency(Tensor tensor, Func<NDimArray,NDimArray> gradF)
        {
            TensorDep = tensor;
            GradFunction = gradF;
        }






        
    }
}
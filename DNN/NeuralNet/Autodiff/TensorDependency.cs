using System;

namespace NeuralNet.Autodiff
{
    /// <summary>
    /// This class represents the dependence of a tensor. This is used to keep track of the tensor's dependency.
    /// For example, if a tensor is made up by the sum of 2 other tensor, this tensor will have 2 TensorDependency object
    /// in it's list of dependency. The gradient function used to compute the gradient of the tensor, with respect to the 
    /// dependencies.
    /// </summary>
    public class TensorDependency
    {

        /// <summary>
        /// The tensor dependence
        /// </summary>
        /// <value></value>
        public Tensor TensorDep { get; set; }

 
        /// <summary>
        /// The gradient function, takes a NDimArray as input, and ouput a NDimArray
        /// The gradient function is used to compute the gradient of the tensor that depends on TensorDep,
        /// with respect to the dependencies.
        /// </summary>
        /// <value></value>
        public Func<NDimArray, NDimArray> GradFunction { get; set; }

        /// <summary>
        /// Constructor used to create a TensorDependency
        /// </summary>
        /// <param name="tensor">The tensor dependence</param>
        /// <param name="gradF">The grad function</param>
        public TensorDependency(Tensor tensor, Func<NDimArray, NDimArray> gradF)
        {
            TensorDep = tensor;
            GradFunction = gradF;
        }

    }
}
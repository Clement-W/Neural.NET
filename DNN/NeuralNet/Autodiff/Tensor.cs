using System;

namespace NeuralNet.Autodiff
{
    public class Tensor
    {
        private NDimArray _data;

        public NDimArray Data
        {
            get
            {
                return _data;
            }
            set
            {
                _data = value;
                // If the data is set mannualy, we need to invalidate the current gradient.
                Grad = null;
            }
        }

        public bool RequiresGrad { get; set; }

        public TensorDependency[] TensorDependencies { get; set; } //TODO: list or array ?

        public Tensor Grad { get; set; }

        public int[] Shape 
        { 
            get
            {
                return _data.Shape;
            } 
        }

        public Tensor(NDimArray data, bool requiresGrad = false, TensorDependency[] dependencies = null)
        {
            _data = new NDimArray(data); //copy to avoid reference issues
            RequiresGrad = requiresGrad;
            TensorDependencies = dependencies;
            Grad = null;

            if (requiresGrad)
            {
                // Set the gradient to 0
                ZeroGrad();
            }
        }

        public override string ToString()
        {
            return $"Tensor, shape=({string.Join(", ",Shape)}, requiresGradient = {RequiresGrad} ,data = ({this.Data})";
        }

        public void ZeroGrad()
        {
            // Create a NDimArray with the same shape as the data, that contains only zeros.
            Grad = new Tensor(NDimArray.Zeros_like(Data));
        }

        // The gradient don't need to be specified if the current tensor is a scalar (one element tensor)
        // Backpropagate a gradient through the graph
        public void Backward(Tensor gradient = null)
        {
            if (RequiresGrad == false)
            {
                throw new InvalidOperationException("Can't call backward method on a tensor that don't requires gradient.");
            }

            if(gradient == null)
            {
                // If the tensor contains only one element
                if(Shape.Length ==1 && Shape[0] == 1)
                {
                    gradient = new Tensor(NDimArray.CreateScalar(1));
                }
                else
                {
                    throw new InvalidOperationException("Gradient argument needs to be specified for a non-scalar tensor");
                }
            }

            // Add the incoming gradient to the current tensor gradient (initialy set to 0)
            Grad.Data = Grad.Data + gradient.Data;

            // Loop recursively into each dependencies of the current tensor to go through the whole graph
            if (TensorDependencies != null)
            {
                foreach (TensorDependency dependency in TensorDependencies)
                {
                    // Compute the gradient with respect to this dependency thanks to the gradient function
                    NDimArray backwardGradient = dependency.GradFunction(gradient.Data);
                    // Backward this gradient through this dependency
                    dependency.TensorDep.Backward(new Tensor(backwardGradient));
                }
            }

        }

        /*public Tensor Sum()
        {
            double data = Data.Sum();
            if (RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {

                }
            }
        }*/

    }
}

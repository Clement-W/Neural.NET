using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    public class Tensor
    {
        private NDimArray _data;

        public NDimArray Data
        {
            get
            {
                return new NDimArray(_data);
            }
            set
            {
                _data = value;
                // If the data is set mannualy, we need to invalidate the current gradient.
                Grad = null;
            }
        }

        public bool RequiresGrad { get; set; }

        public TensorDependency[] TensorDependencies { get; set; }

        public Tensor Grad { get; set; }

        public int[] Shape
        {
            get
            {
                return _data.Shape;
            }
        }

        public int NDim
        {
            get
            {
                return _data.Ndim;
            }
        }

        public int NbElements
        {
            get
            {
                return _data.NbElements;
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

        public Tensor(int[] shape, bool requiresGrad = false, TensorDependency[] dependencies = null)
        : this(new NDimArray(shape), requiresGrad, dependencies) { }


        public Tensor(bool requiresGrad, params double[] data)
        : this(new NDimArray(data), requiresGrad) { }

        public Tensor(bool requiresGrad, int[] shape, params double[] data)
        : this(new NDimArray(shape, data), requiresGrad) { }

        public Tensor(params double[] data)
        : this(new NDimArray(data)) { }

        public override string ToString()
        {
            return $"Tensor, shape=({string.Join(", ", Shape)}), requiresGradient = {RequiresGrad} ,data = ({this.Data})";
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

            //Console.WriteLine("Current tensor is " + this.ToString());
            //Console.WriteLine("AA");
            //Console.WriteLine(string.Join(", ",this.Shape));
            if (RequiresGrad == false)
            {
                throw new InvalidOperationException("Can't call backward method on a tensor that don't requires gradient.");
            }

            if (gradient == null)
            {
                // If the tensor contains only one element
                if (Shape.Length == 1 && Shape[0] == 1)
                {
                    gradient = new Tensor(1);
                }
                else
                {
                    throw new InvalidOperationException("Gradient argument needs to be specified for a non-scalar tensor");
                }
            }

            //Console.WriteLine("received grad = " + string.Join(", ",gradient.Shape) + " " + string.Join(", ", gradient.Data));


            if (Grad == null)
            {
                throw new NullReferenceException($"The gradient is null for this tensor ({this.ToString()}). Maybe the data has been set mannually, which invalidate the gradient.");
            }

            // Add the incoming gradient to the current tensor gradient (initialy set to 0)
            // This allow gadient accumulation
            Grad.Data = Grad.Data + gradient.Data;

            //Console.WriteLine("received grad after grad update = " + string.Join(", ",gradient.Shape) + " " + string.Join(", ", gradient.Data));

            // Loop recursively into each dependencies of the current tensor to go through the whole graph

            if (TensorDependencies != null)
            {
                foreach (TensorDependency dependency in TensorDependencies)
                {
                    //Console.WriteLine("dependency of this tensor : " + dependency.TensorDep.ToString()); 
                    //Console.WriteLine("grad before dependency.gradFunction  = " + string.Join(", ", gradient.Shape) + " " + string.Join(", ", gradient.Data));
                    //Console.WriteLine("oui");
                    NDimArray backwardGradient = dependency.GradFunction(gradient.Data);
                    //Console.WriteLine("BackwardGradient is " + backwardGradient.ToString());
                    //Console.WriteLine("grad after dependency.gradFunction  = " + string.Join(", ", gradient.Shape) + " " + string.Join(", ", gradient.Data));
                    //Console.WriteLine("grad shape : " + string.Join(", ", gradient.Shape));
                    //Console.WriteLine("grad data : " + gradient.Data);
                    //Console.WriteLine("dependency shape : " + string.Join(", ", dependency.TensorDep.Shape));
                    //Console.WriteLine("dependency data : " + string.Join(", ", dependency.TensorDep.Data));

                    // Compute the gradient with respect to this dependency thanks to the gradient function

                    //Console.WriteLine("backwardGrad shape: " + string.Join(", ", backwardGradient.Shape));
                    //Console.WriteLine("backwardGrad data: " + string.Join(", ", backwardGradient.DataArray));
                    // Backward this gradient through this dependency
                    dependency.TensorDep.Backward(new Tensor(backwardGradient));
                }
            }

        }

        // Return the sum of the tensor's elements
        public Tensor Sum()
        {
            NDimArray data = Data.Sum();
            TensorDependency[] dependencies = null;

            if (RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    /*
                    incomingGrad is a one element tensor, because the output of the sum is a 
                    one element tensor. In the sum function, each element has the same weight 
                    (1*x1 + 1*x2 + ... + 1*xn), so the gradient of this tensor wrt to the sum tensor
                    is a tensor of composed of ones, with the same shape of the original tensor.
                    d(grad)/d(thisTensor) = d(grad)/d(sum) * d(sum)/d(thisTensor) = grad * (1,1,1,...)
                    */
                    return incomingGrad * NDimArray.Ones_like(this.Data);
                }
                dependencies = new TensorDependency[] { new TensorDependency(this, GradientFunction) };
            }

            return new Tensor(data, RequiresGrad, dependencies);
        }

        // This method can be used for every operation that uses broadcasting
        // It sums out the broadcasted shape to count them in the gradient
        private static NDimArray HandleBroadcasting(NDimArray gradient, Tensor tensor)
        {

            //Console.WriteLine(string.Join(", ", tensor.Shape));
            //Console.WriteLine(string.Join(", ", gradient.Shape));
            //Console.WriteLine(string.Join(", ", gradient.DataArray));

            // First, sum out the dims added by the broadcast operation, so that the gradient
            // has the same dimensions of the tensor
            // This will handle this example : [[1,2],[3,4]] + [2,2] = [[3,4],[5,6]]
            // If nbDimsAdded is positive, t1 is smaller than the gradient, so t1 has been added multiple times along one or multiple dimensions
            int nbDimsAdded = gradient.Ndim - tensor.Data.Ndim;
            // Sum the gradient's values on the first axis (To take into consideration the dimensions added by broadcasting)
            for (int i = 0; i < nbDimsAdded; i++)
            {
                //Console.WriteLine("dim added");
                gradient = gradient.Sum(axis: 0);
            }

            // Now, to deal with this case :  [[1,2],[3,4]] + [[2,2]] = [[3,4],[5,6]]
            // where the dimensions are broadcasted but not added, we'll need to sum the dims
            // broadcasted by keeping the dimensions

            //Console.WriteLine($"grad before broadcast with keep dims : {string.Join(", ", gradient.Shape)} ; {gradient}");
            //Console.WriteLine(string.Join(", ",tensor.Data.DataArray));

            for (int i = 0; i < tensor.NDim; i++)
            {
                // If the dimension is equal to 1, it means that the operation is broadcasted along this axis
                // If it's a scalar, it doesn't change anything 
                if (tensor.Shape[i] == 1)
                {
                    //Console.WriteLine("opertion will be broadcsted alongo axis " + i);
                    gradient = gradient.Sum(axis: i, keepDims: true);
                    //Console.WriteLine($"grad after sum(axis={i}) : {string.Join(", ", gradient.Shape)} ; {gradient}");
                }
            }
            return gradient;

        }

        public static Tensor operator +(Tensor t1, Tensor t2)
        {
            NDimArray data = t1.Data + t2.Data;
            bool requiresGradient = t1.RequiresGrad || t2.RequiresGrad;

            int nbDependencies = new[] { t1.RequiresGrad, t2.RequiresGrad }.Count(x => x);
            TensorDependency[] dependencies = (nbDependencies > 0) ? new TensorDependency[nbDependencies] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    d(t1+t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1.
                    We also need to handle broadcasting.
                    */

                    return HandleBroadcasting(incomingGrad, t1);
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }
            if (t2.RequiresGrad)
            {
                NDimArray GradientFunction2(NDimArray incomingGrad)
                {
                    /*
                    d(t1+t2)/d(t2) = 1, so we just need to multiply the incoming gradient by 1.
                    We also need to handle broadcasting operation.
                    */
                    return HandleBroadcasting(incomingGrad, t2);
                }
                //nbDependencies-1 = 0 or 1
                dependencies[nbDependencies - 1] = new TensorDependency(t2, GradientFunction2);
            }
            return new Tensor(data, requiresGradient, dependencies);
        }

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            NDimArray data = t1.Data * t2.Data;
            bool requiresGradient = t1.RequiresGrad || t2.RequiresGrad;

            int nbDependencies = new[] { t1.RequiresGrad, t2.RequiresGrad }.Count(x => x);
            TensorDependency[] dependencies = (nbDependencies > 0) ? new TensorDependency[nbDependencies] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    d(t1*t2)/d(t1) = t2, so we just need to multiply the incoming gradient by t2.
                    We also need to handle broadcasting operation.
                    */
                    incomingGrad = incomingGrad * t2.Data;
                    return HandleBroadcasting(incomingGrad, t1);
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }
            if (t2.RequiresGrad)
            {
                NDimArray GradientFunction2(NDimArray incomingGrad)
                {
                    /*
                    d(t1*t2)/d(t2) = t1, so we just need to multiply the incoming gradient by t1.
                    We also need to handle broadcasting operation.
                    */
                    incomingGrad = incomingGrad * t1.Data;
                    return HandleBroadcasting(incomingGrad, t2);
                }
                //nbDependencies-1 = 0 or 1
                dependencies[nbDependencies - 1] = new TensorDependency(t2, GradientFunction2);
            }
            return new Tensor(data, requiresGradient, dependencies);
        }


        public static Tensor operator -(Tensor t1)
        {
            NDimArray data = -t1.Data;
            bool requiresGradient = t1.RequiresGrad;
            TensorDependency[] dependencies = (requiresGradient) ? new TensorDependency[1] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    d(-t1)/d(t1) = -1, so we just need to multiply the incoming gradient by -1.
                    */
                    return -incomingGrad;
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }

            return new Tensor(data, requiresGradient, dependencies);
        }

        public static Tensor operator -(Tensor t1, Tensor t2)
        {
            NDimArray data = t1.Data - t2.Data;
            bool requiresGradient = t1.RequiresGrad || t2.RequiresGrad;

            int nbDependencies = new[] { t1.RequiresGrad, t2.RequiresGrad }.Count(x => x);
            TensorDependency[] dependencies = (nbDependencies > 0) ? new TensorDependency[nbDependencies] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    d(t1-t2)/d(t1) = 1, so we just need to multiply the incoming gradient by 1.
                    We also need to handle broadcasting operation.
                    */
                    return HandleBroadcasting(incomingGrad, t1);
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }
            if (t2.RequiresGrad)
            {
                NDimArray GradientFunction2(NDimArray incomingGrad)
                {
                    /*
                    d(t1*t2)/d(t2) = -1, so we just need to multiply the incoming gradient by -1.
                    We also need to handle broadcasting operation.
                    */
                    return HandleBroadcasting(-incomingGrad, t2);
                }
                //nbDependencies-1 = 0 or 1
                dependencies[nbDependencies - 1] = new TensorDependency(t2, GradientFunction2);
            }
            return new Tensor(data, requiresGradient, dependencies);
        }


        public static Tensor operator /(Tensor t1, Tensor t2)
        {
            NDimArray data = t1.Data / t2.Data;
            bool requiresGradient = t1.RequiresGrad || t2.RequiresGrad;

            int nbDependencies = new[] { t1.RequiresGrad, t2.RequiresGrad }.Count(x => x);
            TensorDependency[] dependencies = (nbDependencies > 0) ? new TensorDependency[nbDependencies] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    d(t1/t2)/d(t1) = 1/t2, so we just need to multiply the incoming gradient by 1/t2.
                    We also need to handle broadcasting operation.
                    */
                    incomingGrad = incomingGrad / t2.Data;
                    return HandleBroadcasting(incomingGrad, t1);
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }
            if (t2.RequiresGrad)
            {
                NDimArray GradientFunction2(NDimArray incomingGrad)
                {
                    /*
                    d(t1/t2)/d(t2) = -t1/(t2*t2), so we just need to multiply the incoming gradient by -t1/(t2*t2).
                    We also need to handle broadcasting operation.
                    */
                    incomingGrad = incomingGrad * (-t1.Data / (t2.Data * t2.Data));
                    return HandleBroadcasting(incomingGrad, t2);
                }
                //nbDependencies-1 = 0 or 1
                dependencies[nbDependencies - 1] = new TensorDependency(t2, GradientFunction2);
            }
            return new Tensor(data, requiresGradient, dependencies);
        }

        public static Tensor Matmul(Tensor t1, Tensor t2)
        {

            NDimArray data = NDimArray.Matmul(t1.Data, t2.Data);
            bool requiresGradient = t1.RequiresGrad || t2.RequiresGrad;

            int nbDependencies = new[] { t1.RequiresGrad, t2.RequiresGrad }.Count(x => x);
            TensorDependency[] dependencies = (nbDependencies > 0) ? new TensorDependency[nbDependencies] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    With t1 (n1,m1), t2 (m1,m2) and t3 = t1@t2 is (n1,m2)
                    So the incoming gradient wrt t3 is (n1,m2)
                    d(t1@t2)/d(t1) = t2
                    So we just need to matmul the incoming gradient by t2. But t2 is (m1,m2)
                    and the incoming gradient is (n1,m2). So we need to do grad @ t2.Transpose
                    */
                    //Console.WriteLine("on passe la dans t1 grad matmul");

                    return NDimArray.Matmul(incomingGrad, t2.Data.Transpose());
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }
            if (t2.RequiresGrad)
            {
                NDimArray GradientFunction2(NDimArray incomingGrad)
                {

                    /*
                    d(t1@t2)/d(t2) = t1
                    So we just need to matmul the incoming gradient by t1. But t1 is (n1,m1 )
                    and the incoming gradient is (n1,m2). So we need to do t1.Transpose @ grad
                    */
                    //Console.WriteLine("on passe la dans t2 grad matmul");
                    //Console.WriteLine("matmul entre : ");
                    //Console.WriteLine("t1.Data : (" + string.Join(", ",t1.Shape) +") " + t1.Data + ".T donc t1.data transposed =  (" + string.Join(", ",t1.Shape) + ") " + t1.Data.Transpose());
                    //Console.WriteLine("et (" + string.Join(", ",incomingGrad.Shape) + ") " + incomingGrad);
                    //Console.WriteLine(" res = (" + string.Join(", ",NDimArray.Matmul(t1.Data.Transpose(), incomingGrad).Shape) + ") ; " +  NDimArray.Matmul(t1.Data.Transpose(), incomingGrad));


                    //Console.WriteLine("INCOMINGgrad avant return = (" + string.Join(", ",incomingGrad.Shape) + ") " + incomingGrad );
                    return NDimArray.Matmul(t1.Data.Transpose(), incomingGrad);
                }
                //nbDependencies-1 = 0 or 1
                dependencies[nbDependencies - 1] = new TensorDependency(t2, GradientFunction2);
            }
            return new Tensor(data, requiresGradient, dependencies);
        }

        public Tensor Slice2DTensor(int start, int end)
        {
            NDimArray data = Data.Slice2DArray(start, end);
            TensorDependency[] dependencies = null;

            if (RequiresGrad)
            {
                NDimArray GradientFunction(NDimArray incomingGrad)
                {
                    /*
                    Copy the gradients of the keeped dims into the new gradient array, but set every
                    gradient that have not been keeped to 0
                    */
 
                    NDimArray newGrad = NDimArray.Zeros_like(this.Data);

                    //Copy the gradient
                    if (end > Shape[0])
                    {
                        end = Shape[0];
                    }
                    for(int i = 0;start<end;i++){
                        for(int j =0;j<incomingGrad.Shape[1];j++){
                            newGrad[start,j] = incomingGrad[i,j];
                        }
                        start++;
                    }
       

                    return newGrad;
                }
                dependencies = new TensorDependency[] { new TensorDependency(this, GradientFunction) };
            }

            return new Tensor(data, RequiresGrad, dependencies);
        }



        //TODO: add pow operator
    }

}

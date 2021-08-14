using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    /// <summary>
    /// This class represents a Tensor, basically it's a ndimarray that supports gradient computation
    /// </summary>
    public class Tensor
    {
        /// <summary>
        /// The data contained in the tensor is a ndim array
        /// </summary>
        private NDimArray _data;

        /// <summary>
        /// Property used to access _data
        /// If the data is mannualy set, the gradient is set to null
        /// </summary>
        /// <value></value>
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

        /// <summary>
        /// Boolean which indicates if the gradient has to be computed for this tensor
        /// </summary>
        /// <value></value>
        public bool RequiresGrad { get; set; }

        /// <summary>
        /// This list contains the tensors on which the current tensor depends
        /// The TensorDependency object is composed of the tensor dependencies, and the gradient functions 
        /// used to compute the gradient of this tensor, with respect to the dependencies
        /// </summary>
        /// <value></value>
        public TensorDependency[] TensorDependencies { get; set; }

        /// <summary>
        /// The computed gradient with respect to this tensor
        /// </summary>
        /// <value></value>
        public Tensor Grad { get; set; }

        /// <summary>
        /// The shape of the tensor
        /// </summary>
        /// <value></value>
        public int[] Shape
        {
            get
            {
                return _data.Shape;
            }
        }

        /// <summary>
        /// The number of dimension of this tensor
        /// </summary>
        /// <value></value>
        public int Ndim
        {
            get
            {
                return _data.Ndim;
            }
        }

        /// <summary>
        /// The number of elements contained in this tensor
        /// </summary>
        /// <value></value>
        public int NbElements
        {
            get
            {
                return _data.NbElements;
            }
        }

        /// <summary>
        /// Constructor used to create a tensor with a ndim array
        /// </summary>
        /// <param name="data">The ndim array that compose the tensor</param>
        /// <param name="requiresGrad">Boolean that indicates if the gradient needs to be computed</param>
        /// <param name="dependencies">The tensor dependencies of this tensor</param>
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

        /// <summary>
        /// Constructor used to create a tensor with the given shapes
        /// </summary>
        /// <param name="shape">The shape of the tensor</param>
        /// <param name="requiresGrad">Boolean that indicates if the gradient needs to be computed</param>
        /// <param name="dependencies">The tensor dependencies of this tensor</param>
        public Tensor(int[] shape, bool requiresGrad = false, TensorDependency[] dependencies = null)
        : this(new NDimArray(shape), requiresGrad, dependencies) { }

        /// <summary>
        /// Constructor used to create a 1dim tensor with the given data 
        /// </summary>
        /// <param name="requiresGrad">Boolean that indicates if the gradient needs to be computed</param>
        /// <param name="data">The data contained in the 1-dim array that compose the tensor</param>
        public Tensor(bool requiresGrad, params double[] data)
        : this(new NDimArray(data), requiresGrad) { }

        /// <summary>
        /// Constructor used to create a tensor with the given shape and the given data
        /// </summary>
        /// <param name="requiresGrad">Boolean that indicates if the gradient needs to be computed</param>
        /// <param name="shape">The shape of the tensor</param>
        /// <param name="data">The data contained in the n-dim array that compose the tensor</param>
        public Tensor(bool requiresGrad, int[] shape, params double[] data)
        : this(new NDimArray(shape, data), requiresGrad) { }

        /// <summary>
        /// Constructor used to create a 1dim tensor with the given data
        /// </summary>
        /// <param name="data">The data contained in the 1-dim array that compose the tensor</param>
        public Tensor(params double[] data)
        : this(new NDimArray(data)) { }

        /// <summary>
        /// Indexer to access the values of the tensor
        /// </summary>
        /// <value></value>
        public double this[params int[] indexes]
        {
            get
            {
                return Data[indexes];
            }
            set
            {
                _data[indexes] = value;
            }
        }

        public override string ToString()
        {
            return $"Tensor, shape=({string.Join(", ", Shape)}), requiresGradient = {RequiresGrad} ,data = ({this.Data})";
        }

        /// <summary>
        /// Set the gradient w.r.t this tensor to 0
        /// </summary>
        public void ZeroGrad()
        {
            // Create a NDimArray with the same shape as the data, that contains only zeros.
            Grad = new Tensor(NDimArray.Zeros_like(Data));
        }



        /// <summary>
        /// Backpropagate a gradient through the auto differenciation graph by
        /// recurcively calling this method on the tensor dependencies.
        /// The gradient don't need to be specified if the current tensor is a scalar (one element tensor) 
        /// </summary>
        /// <param name="gradient">The gradient that will be backpropagated through the grpah</param>
        public void Backward(Tensor gradient = null)
        {

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


            if (Grad == null)
            {
                throw new NullReferenceException($"The gradient is null for this tensor ({this.ToString()}). Maybe the data has been set mannually, which invalidate the gradient.");
            }

            // Add the incoming gradient to the current tensor gradient (initialy set to 0)
            // This allow gadient accumulation
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

        /// <summary>
        /// Return the sum of the tensor's elements
        /// </summary>
        /// <returns></returns>
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


        /// <summary>
        /// This method is used for every ops that uses broadcasting. It sums out the broadcasted shape to 
        /// count them in the gradient.
        /// </summary>
        /// <param name="gradient">The incoming gradient</param>
        /// <param name="tensor">The tensor from which the gradient is computed</param>
        /// <returns></returns>
        private static NDimArray HandleBroadcasting(NDimArray gradient, Tensor tensor)
        {

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
            // where the operation is broadcasted but no dimension is added, we'll need to sum the 
            // broadcasted dims  by keeping the dimensions.

            // For each dimension
            for (int i = 0; i < tensor.Ndim; i++)
            {
                // If the dimension is equal to 1, it means that the operation is broadcasted along this axis
                // If it's a scalar, it doesn't change anything 
                if (tensor.Shape[i] == 1)
                {
                    gradient = gradient.Sum(axis: i, keepDims: true);
                }
            }
            return gradient;

        }

        /// <summary>
        /// + operator for tensors to support addition between 2 tensors
        /// This method will add the 2 ndim array contained in the tensor, and then if one of the
        /// two tensors requires gradient computation, the result of t1+t2 will also requires gradient computation.
        /// So we add t1 and t2 as dependencies of this tensor, with the corresponding gradient functionss.
        /// </summary>
        /// <param name="t1">The first tensor</param>
        /// <param name="t2">The second tensor</param>
        /// <returns>A new tensor equal to t1+t2</returns>
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

        /// <summary>
        /// + operator to support addition between a scalar and a tensor
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="t2">The tensor</param>
        /// <returns>A new tensor equal to scalar + t2</returns>
        public static Tensor operator +(double scalar, Tensor t2)
        {
            return new Tensor(scalar) + t2;
        }

        /// <summary>
        /// + operator to support addition between a tensor and a scalar
        /// </summary>
        /// <param name="t1">The tensor</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>A new tensor equal to t1 + scalar</returns>
        public static Tensor operator +(Tensor t1, double scalar)
        {
            return t1 + new Tensor(scalar);
        }

        /// <summary>
        /// * operator for tensors to support multiplication between 2 tensors
        /// This method will multiply the 2 ndim array contained in the tensor, and then if one of the
        /// two tensors requires gradient computation, the result of t1*t2 will also requires gradient computation.
        /// So we add t1 and t2 as dependencies of this tensor, with the corresponding gradient functions.
        /// </summary>
        /// <param name="t1">The first tensor</param>
        /// <param name="t2">The second tensor</param>
        /// <returns>A new tensor equal to t1*t2</returns>
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

        /// <summary>
        /// * operator to support multiplication between a scalar and a tensor
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="t2">The tensor</param>
        /// <returns>A new tensor equal to scalar * t2</returns>
        public static Tensor operator *(double scalar, Tensor t2)
        {
            return new Tensor(scalar) * t2;
        }

        /// <summary>
        /// * operator to support multiplication between a tensor and a scalar
        /// </summary>
        /// <param name="t1">The tensor</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>A new tensor equal to t1 * scalar</returns>
        public static Tensor operator *(Tensor t1, double scalar)
        {
            return t1 * new Tensor(scalar);
        }

        /// <summary>
        /// - operator to negate the tensor.
        /// This method will negate the ndim array contained in the tensor, and then if the
        /// original tensor requires gradient computation, the result of -t1 will also requires gradient computation.
        /// So we add t1 as a dependency of this tensor, with the corresponding gradient functions.
        /// </summary>
        /// <param name="t1">The tensor</param>
        /// <returns>A new tensor equal to -t1</returns>
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

        /// <summary>
        /// - operator for tensors to support substraction between 2 tensors
        /// This method will substract the 2 ndim array contained in the tensor, and then if one of the
        /// two tensors requires gradient computation, the result of t1-t2 will also requires gradient computation.
        /// So we add t1 and t2 as dependencies of this tensor, with the corresponding gradient functions.
        /// </summary>
        /// <param name="t1">The first tensor</param>
        /// <param name="t2">The second tensor</param>
        /// <returns>A new tensor equal to t1-t2</returns>
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

        /// <summary>
        /// - operator to support substraction between a scalar and a tensor
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="t2">The tensor</param>
        /// <returns>A new tensor equal to scalar - t2</returns>
        public static Tensor operator -(double scalar, Tensor t2)
        {
            return new Tensor(scalar) - t2;
        }

        /// <summary>
        /// - operator to support substraction between a tensor and a scalar
        /// </summary>
        /// <param name="t1">The tensor</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>A new tensor equal to t1 - scalar</returns>
        public static Tensor operator -(Tensor t1, double scalar)
        {
            return t1 - scalar;
        }


        /// <summary>
        /// / operator for tensors to support true division between 2 tensors
        /// This method will perform true division between the 2 ndim array contained in the tensor, 
        /// and then if one of the two tensors requires gradient computation, the result of t1-t2 will 
        /// also requires gradient computation.
        /// So we add t1 and t2 as dependencies of this tensor, with the corresponding gradient functionss.
        /// </summary>
        /// <param name="t1">The first tensor</param>
        /// <param name="t2">The second tensor</param>
        /// <returns>A new tensor equal to t1/t2</returns>
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

        /// <summary>
        /// / operator to support true division between a scalar and a tensor
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="t2">The tensor</param>
        /// <returns>A new tensor equal to scalar / t2</returns>
        public static Tensor operator /(double scalar, Tensor t2)
        {
            return new Tensor(scalar) / t2;
        }

        /// <summary>
        /// / operator to support substraction between a tensor and a scalar
        /// </summary>
        /// <param name="t1">The tensor</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>A new tensor equal to t1 / scalar</returns>
        public static Tensor operator /(Tensor t1, double scalar)
        {
            return t1 * new Tensor(scalar);
        }

     
        /// <summary>
        /// Perform log base e (ln) on the tensor. If t1 requires gradient computation, store t1 as
        /// a dependency.
        /// </summary>
        /// <param name="t1">The tensor</param>
        /// <returns>A new tensor equal to ln(t1)</returns>
        public static Tensor Log(Tensor t1)
        {
            NDimArray data = NDimArray.Log(t1.Data);
            bool requiresGradient = t1.RequiresGrad;
            TensorDependency[] dependencies = (requiresGradient) ? new TensorDependency[1] : null;


            if (t1.RequiresGrad)
            {
                NDimArray GradientFunction1(NDimArray incomingGrad)
                {
                    /*
                    d(ln(t1))/d(t1) = 1/t1, so we just need to multiply the incoming gradient by 1/t1.
                    */
                    return incomingGrad * (1 / t1.Data);
                }
                dependencies[0] = new TensorDependency(t1, GradientFunction1);
            }

            return new Tensor(data, requiresGradient, dependencies);
        }

        /// <summary>
        /// Perform matrix multiplication between 2 tensors. If one of the two tensor requires grad computation,
        /// store them into the result's dependencies.
        /// </summary>
        /// <param name="t1">The first tensor</param>
        /// <param name="t2">The second tensor</param>
        /// <returns></returns>
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

        /// <summary>
        /// Slice the data of a 2D tensor. If the tensor requires gradient computation,
        /// it adds the current tensor as a dependency.
        /// </summary>
        /// <param name="start">Start index</param>
        /// <param name="end">End index</param>
        /// <returns>A subpart of the original tensor</returns>
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
                    for (int i = 0; start < end; i++)
                    {
                        for (int j = 0; j < incomingGrad.Shape[1]; j++)
                        {
                            newGrad[start, j] = incomingGrad[i, j];
                        }
                        start++;
                    }


                    return newGrad;
                }
                dependencies = new TensorDependency[] { new TensorDependency(new Tensor(this.Data,true), GradientFunction) };
            }

            return new Tensor(data, RequiresGrad, dependencies);
        }

        /// <summary>
        /// Return the indexes of the max values along the rows of a 2D array. 
        /// This method should not be there, it's temporary since the max function is not implemented.
        /// </summary>
        /// <returns></returns>
        public int[] GetPredictionsIndexes()
        {
            if (Ndim != 2)
            {
                throw new NotImplementedException("This method is not yet implemented yet for other dim than 2Dim arrays");
            }

            // Return the list of max indexes along each column (the reduced dim corresponds to the rows)
            int[] res = new int[Shape[0]];
            double max = Data.DataArray[0];
            int maxIndex = 0;
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    if (this[i, j] > max)
                    {
                        max = this[i, j];
                        maxIndex = j;
                    }
                }
                res[i] = maxIndex;

                max = double.MinValue;
                maxIndex = -1;
            }

            return res;

        }


        //TODO: add pow operator
    }

}

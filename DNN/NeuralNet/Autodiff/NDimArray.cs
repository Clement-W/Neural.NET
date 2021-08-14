using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    /// <summary>
    /// This class represents a n-dimentional array
    /// </summary>
    public class NDimArray
    {

        // The data contained in the n-dim array is stored in a one-dim array 
        public double[] DataArray { get; private set; }

        // The shape of the n-dim array, for example it would be [5,3] for a (5x3) 2-dim array 
        private int[] _shape;

        // The list of indexes used to access the one dimensional array that contains the n-dim array data
        public int[] StepIndexes { get; private set; }

        // The number of dimentions of the n-dim array
        public int Ndim
        {
            get
            {
                return Shape.Length;
            }
        }

        // The property used to get and set shape_
        public int[] Shape
        {
            get
            {
                if (_shape != null)
                {
                    return (int[])_shape.Clone();
                }
                else
                {
                    return null;
                }
            }
            set
            {
                // If no shape has been set, or the new shape is compatible with the old one
                if (NbElements == 0 || NbElements == value.Aggregate(1, (a, b) => a * b))
                {
                    _shape = value;
                    InitStepIndexes();
                }
                else
                {
                    throw new ArgumentException("The new shape doesn't fit the current array.");
                }
            }
        }



        // Return the number of elements contained in the ndim array
        public int NbElements
        {
            get
            {
                if (Shape == null)
                {
                    return 0;
                }
                else
                {
                    if (DataArray == null)
                    {
                        if (Shape.Length == 0)
                        {
                            return 0;
                        }
                        else
                        {
                            // If the data array is null, return the future data length by multitplying every shapes
                            return Shape.Aggregate(1, (a, b) => a * b);
                        }
                    }
                    else
                    {
                        return DataArray.Length;
                    }
                }
            }
        }

        /// <summary>
        /// Constructor used to create a n-dim array of the given shape
        /// </summary>
        /// <param name="shape">The shape of the ndim array</param>
        public NDimArray(int[] shape)
        {
            Shape = shape;
            DataArray = new double[NbElements];
        }

        /// <summary>
        /// Constructor used to create a n-dim array with the given shape, filled by the given data
        /// </summary>
        /// <param name="shape">The shape of the ndim array</param>
        /// <param name="data">The data contained in the ndim array</param>
        public NDimArray(int[] shape, params double[] data)
        {
            Shape = shape;
            if (NbElements == data.Length)
            {
                DataArray = data;
            }
            else
            {
                throw new ArgumentException("The given shape, and array are not compatible.");
            }

        }

        /// <summary>
        /// Constructor used to create a ndim array by copying the data and the shape of another ndim array
        /// </summary>
        /// <param name="arr">The ndim array that will be copied</param>
        public NDimArray(NDimArray arr)
        {
            Shape = arr.Shape;
            DataArray = arr.DataArray;
        }

        // Used to create a 1DimArray
        /// <summary>
        /// Constructor used to create a 1dim array that contains the given data
        /// </summary>
        /// <param name="data">The data contained in the ndim array </param>
        public NDimArray(params double[] data)
        {
            Shape = new int[] { data.Length };
            DataArray = data;
        }


        public override string ToString()
        {
            return $"NDimArray of shape ({string.Join(", ", Shape)}), data=[{string.Join(", ", DataArray)}]";
        }

        /// <summary>
        /// Initialize the StepIndexes property with the step indexes adapted to the shape
        /// </summary>
        private void InitStepIndexes()
        {
            int[] steps = new int[this.Ndim];

            int step = 1;
            for (int i = Ndim - 1; i >= 0; i--)
            {
                steps[i] = step;
                step *= Shape[i];
            }
            StepIndexes = steps;
        }


        /// <summary>
        /// Fill an array with the given value
        /// </summary>
        /// <param name="value">The value that will compose the ndim array</param>
        public void FillWithValue(double value)
        {
            for (int i = 0; i < NbElements; i++)
            {
                DataArray[i] = value;
            }
        }

        /// <summary>
        /// Convert a list of indexes like [2,3] to the corresponding index in the one-dim array that 
        /// contains the data
        /// </summary>
        /// <param name="indexes">The indexes used to access the ndim array</param>
        /// <returns>An index that can be used to access the value in the 1dim array DataArray</returns>
        private int ConvertNDIndexTo1DIndex(int[] indexes)
        {
            if (indexes.Length == Ndim)
            {
                int index = 0;
                for (int i = 0; i < Ndim; i++)
                {
                    index += indexes[i] * StepIndexes[i];
                }
                return index;
            }
            else
            {
                throw new ArgumentException("The number of indexes given (" + indexes.Length + ")" + " doesn't fit the array shape (" + Ndim + ").");
            }
        }

        /// <summary>
        /// Indexer of the ndim array
        /// </summary>
        /// <value>The value contained in the ndim array</value>
        public double this[params int[] indexes]
        {
            get
            {
                return DataArray[ConvertNDIndexTo1DIndex(indexes)];
            }
            set
            {
                DataArray[ConvertNDIndexTo1DIndex(indexes)] = value;
            }
        }

        /// <summary>
        /// Print the ndim array as a matrix if it's a 2dim array
        /// </summary>
        public void PrintAsMatrix()
        {
            if (Ndim == 2)
            {
                for (int i = 0; i < Shape[0]; i++)
                {
                    for (int j = 0; j < Shape[1]; j++)
                    {
                        Console.Write(this[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            else
            {
                throw new InvalidOperationException("Can't print this n-darray as a matrix, it's shape is " + string.Join(", ", Shape));
            }
        }

        /// <summary>
        /// Create an array with the same shape of the given array, filled with zeros
        /// </summary>
        /// <param name="arr">The array from which the shapes will be copied</param>
        /// <returns>A ndim array filled with zeros</returns>
        public static NDimArray Zeros_like(NDimArray arr)
        {
            return new NDimArray(arr.Shape);
        }

        /// <summary>
        /// Create an array with the same shape of the given array, filled with ones
        /// </summary>
        /// <param name="arr">The array from which the shapes will be copied</param>
        /// <returns>A ndim array filled with ones</returns>
        public static NDimArray Ones_like(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            res.FillWithValue(1);
            return res;
        }

        /// <summary>
        /// Create an ndim array with the given shape, with randomized values
        /// </summary>
        /// <param name="shape">The shape of the ndim array</param>
        /// <returns>A randomized ndim array</returns>
        public static NDimArray Random(params int[] shape)
        {
            Random rd = new Random();
            NDimArray res = new NDimArray(shape);
            for (int i = 0; i < res.NbElements; i++)
            {
                res.DataArray[i] = rd.NextDouble();
            }
            return res;
        }

        /// <summary>
        /// Create an ndim array with the given shape, with randomized values > inf and < sup
        /// </summary>
        /// <param name="shape">The shape of the ndim array</param>
        /// <param name="inf">The lower limit of the randomized values</param>
        /// <param name="sup">The upper limit of the randomized values</param>
        /// <returns>A randomized ndim array</returns>
        public static NDimArray Random(int[] shape, double inf, double sup)
        {
            Random rd = new Random();
            NDimArray res = new NDimArray(shape);
            for (int i = 0; i < res.NbElements; i++)
            {
                // Generate random number between inf and sup
                res.DataArray[i] = rd.NextDouble() * (sup - inf) + inf;
            }
            return res;
        }

        /// <summary>
        /// Shuffle the data of the current array
        /// </summary>
        public void Shuffle()
        {
            Random rd = new Random();
            int nbShuffled = this.DataArray.Length;
            int index;
            double tmp;
            while (nbShuffled > 1)
            {
                index = rd.Next(nbShuffled--);
                tmp = DataArray[nbShuffled];
                DataArray[nbShuffled] = DataArray[index];
                DataArray[index] = tmp;
            }
        }


        // Sum the element of an array, or sum along the given axis
        //TODO: broadcasting is not supported for more than 2D arrays,
        // so this function doesn't support axis >=2 (only null, 0 or 1)

        /// <summary>
        /// Sum the elements of an array, or sum the elements along the given axis
        /// TODO: broadcasting is not supported for nDim arrays with n > 2, that's why this function
        ///  doesn't support axis >=2 (only null, 0 or 1)
        /// </summary>
        /// <param name="axis">The axis along which the values are summed</param>
        /// <param name="keepDims">To tell if the dimensions of the original array has to be keeped or not</param>
        /// <returns>A new ndim array</returns>
        public NDimArray Sum(int? axis = null, bool keepDims = false)
        {
            // Return the sum of the elements
            if (axis == null || (Ndim == 2 && ((Shape[0] == 1 && axis == 1) || (Shape[1] == 1 && axis == 0))))
            {
                if (!keepDims)
                {
                    return new NDimArray(this.DataArray.Sum());
                }
                else
                {
                    // Return the same result, but with the same dimension
                    int[] newShape = new int[Ndim];
                    Array.Fill(newShape, 1);
                    return new NDimArray(newShape, new double[] { this.DataArray.Sum() });
                }
            }
            // Return the row, without a dimension if keepDims is false
            else if (Ndim == 2 && ((Shape[0] == 1 && axis == 0) || (Shape[1] == 1 && axis == 1)))
            {
                if (!keepDims)
                {
                    return new NDimArray(new int[] { NbElements }, DataArray);
                }
                else
                {
                    return new NDimArray(this);
                }
            }
            // Sum the element along the rows (delete the first dimension)
            // if shapes are (a,b), then the sum(axis:0) will return a ndimarray with shape (b) 
            else if (Ndim == 2 && axis == 0)
            {
                int[] newShape;
                if (!keepDims)
                {
                    newShape = new int[] { Shape[1] };
                }
                else
                {
                    newShape = new int[] { 1, Shape[1] };
                }

                double[] newData = new double[Shape[1]];
                for (int j = 0; j < Shape[1]; j++)
                {
                    for (int i = 0; i < Shape[0]; i++)
                    {
                        newData[j] += this[i, j];
                    }
                }
                return new NDimArray(newShape, newData);

            }
            // Sum the element along the columns (delete the second dimension)
            // if shapes are (a,b), then the sum(axis:1) will return a ndimarray with shape (a)
            else if (Ndim == 2 && axis == 1)
            {
                int[] newShape;
                if (!keepDims)
                {
                    newShape = new int[] { Shape[0] };
                }
                else
                {
                    newShape = new int[] { Shape[0], 1 };
                }

                double[] newData = new double[Shape[0]];
                for (int i = 0; i < Shape[0]; i++)
                {
                    for (int j = 0; j < Shape[1]; j++)
                    {
                        newData[i] += this[i, j];
                    }
                }
                return new NDimArray(newShape, newData);
            }
            else if (Ndim == 1 && axis == 0)
            {
                return new NDimArray(this.DataArray.Sum());
            }

            else if (axis > Ndim - 1)
            {
                throw new InvalidOperationException($"Can't sum the array along axis {axis}, this array has only {Ndim} dimensions.");
            }
            else
            {
                throw new NotImplementedException($"Sum with a specified axis is not yet supported for ndim>2");
            }
        }



        /// <summary>
        /// Perform the Tanh function on the DataArray of the given array
        /// </summary>
        /// <param name="arr">The given array on which Tanh will be performed</param>
        /// <returns>A new ndim array</returns>
        public static NDimArray Tanh(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            for (int i = 0; i < arr.NbElements; i++)
            {
                res.DataArray[i] = Math.Tanh(arr.DataArray[i]);
            }
            return res;
        }


        /// <summary>
        /// Perform the exp function on the DataArray of the given array
        /// </summary>
        /// <param name="arr">The given array on which exp will be performed</param>
        /// <returns>A new ndim array</returns>
        public static NDimArray Exp(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            for (int i = 0; i < arr.NbElements; i++)
            {
                res.DataArray[i] = Math.Exp(arr.DataArray[i]);
            }
            return res;
        }

        /// <summary>
        /// Perform the LeakyRelu function on the DataArray of the given array
        /// </summary>
        /// <param name="arr">The given array on which LeakyRelu will be performed</param>
        /// <returns>A new ndim array</returns>
        public static NDimArray LeakyRelu(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            for (int i = 0; i < arr.NbElements; i++)
            {
                if (arr.DataArray[i] >= 0)
                {
                    res.DataArray[i] = arr.DataArray[i];
                }
                else
                {
                    res.DataArray[i] = 0.01 * arr.DataArray[i];
                }

            }
            return res;
        }

        /// <summary>
        /// Transpose a 2dim array
        /// //TODO: support n-dim transpose, (currently not supported for other than 2dim array)
        /// </summary>
        /// <returns>The transposed current array</returns>
        public NDimArray Transpose()
        {

            if (Ndim == 1)
            {
                return new NDimArray(this);
            }
            else if (Ndim == 2)
            {
                NDimArray res = new NDimArray(new int[] { Shape[1], Shape[0] });
                for (int newCol = 0; newCol < res.Shape[0]; newCol++)
                {
                    for (int newRow = 0; newRow < res.Shape[1]; newRow++)
                    {
                        res[newCol, newRow] = this[newRow, newCol];
                    }
                }
                return res;
            }
            else
            {
                throw new NotImplementedException("Can't transpose a ndimarray with n > 2 (TODO)");
            }
        }


        /// <summary>
        /// Return a subpart of the 2D array by returning the rows from start to end given indexes
        /// </summary>
        /// <param name="start">The index of the row where the slice beggins</param>
        /// <param name="end">The index of the row where the slice ends</param>
        /// <returns> A subpart of the 2Darray as a new ndim array </returns>
        public NDimArray Slice2DArray(int start, int end)
        {
            if (Ndim > 2)
            {
                throw new NotImplementedException("NDimARRAY slice is not implemented yet for more than 2D arrays");
            }
            if (start < 0 || start > Shape[0] || end < 0)
            {
                throw new InvalidOperationException($"The start index should be in [0,{Shape[0]}] and the end index should be > 0. The given start is {start} and the given end is {end}. ");
            }
            if (end - start <= 0)
            {
                throw new InvalidOperationException($"Can't slice NDimArray, from {start} to {end} because {end} <= {start}");
            }

            //slice the array on the first dimension
            if (end > Shape[0])
            {
                end = Shape[0];
            }

            int[] newShape = Shape;
            newShape[0] = end - start;

            NDimArray res = new NDimArray(newShape);
            for (int i = 0; start < end; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    res[i, j] = this[start, j];
                }
                start++;
            }

            return res;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval in a 1Dim array.
        /// </summary>
        /// <param name="start">The start value</param>
        /// <param name="end">The end value</param>
        /// <param name="step">The space between values</param>
        /// <returns>A 1dim array</returns>
        public static NDimArray Arange(int start, int end, int step = 1)
        {
            if (start < 0 || start > end || end < 0)
            {
                throw new ArgumentException($"Given start index ({start}) must be >=0 and < end, and the end index ({end}) must be > 0");
            }
            if (end == start)
            {
                return new NDimArray(new int[] { });
            }
            if (step > end - start)
            {
                return new NDimArray(start);
            }

            // (end - start)/step rounded up
            int nbValues = Convert.ToInt16(Math.Ceiling((double)(end - start) / (double)step));

            NDimArray res = new NDimArray(new int[] { nbValues });

            int val = start - step;
            for (int i = 0; i < nbValues; i++)
            {
                val = val + step;
                res.DataArray[i] = val;
            }

            return res;
        }



        /// <summary>
        /// Check if two shapes are compatible for broadcast. In a shape, for each dimension, they are 
        /// compatible when they're equal, or one of them is 1.
        /// https://numpy.org/doc/stable/user/basics.broadcasting.html
        /// </summary>
        /// <param name="shape1"></param>
        /// <param name="shape2"></param>
        /// <returns></returns>
        public static bool IsShapesBroadcastable(int[] shape1, int[] shape2)
        {
            shape1 = Enumerable.Reverse(shape1).ToArray();
            shape2 = Enumerable.Reverse(shape2).ToArray();
            int minShapeLength = (shape1.Length <= shape2.Length) ? shape1.Length : shape2.Length;
            for (int i = minShapeLength - 1; i >= 0; i--)
            {
                if (shape1[i] != 1 && shape2[i] != 1 && shape1[i] != shape2[i])
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Apply the given operation between two ndim array of the same shape
        /// </summary>
        /// <param name="operation"> The operation that takes 2 doubles and return a double</param>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns></returns>
        private static NDimArray ApplyOperationBetweenNDimArray(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2)
        {
            NDimArray res = new NDimArray(arr1.Shape);
            for (int i = 0; i < arr1.NbElements; i++)
            {
                res.DataArray[i] = operation(arr1.DataArray[i], arr2.DataArray[i]);
            }
            return res;
        }

        /// <summary>
        /// Apply the given operation between a ndim array and a ndim array that contains only one value (a scalar)
        /// </summary>
        /// <param name="operation">The operation that takes 2 doubles and return a double</param>
        /// <param name="arr1">The first array (can be the scalar)</param>
        /// <param name="arr2">The second array (can be the scalar)</param>
        /// <returns></returns>
        private static NDimArray ApplyOperationWithScalar(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2)
        {
            // Know which one is the scalar or the array
            NDimArray array = (arr1.Ndim == 1 && arr1.Shape[0] == 1) ? arr2 : arr1;

            NDimArray res = new NDimArray(array.Shape);
            for (int i = 0; i < array.NbElements; i++)
            {
                if (array == arr1)
                {
                    res.DataArray[i] = operation(arr1.DataArray[i], arr2.DataArray[0]);
                }
                else
                {
                    res.DataArray[i] = operation(arr1.DataArray[0], arr2.DataArray[i]);
                }
            }
            return res;
        }



        /// <summary>
        /// Return the shapes of an array after broadcsting the shapes of two ndim array
        /// </summary>
        /// <param name="shapeArr1">The shape of the first array</param>
        /// <param name="shapeArr2">The shape of the second array</param>
        /// <returns>The shape of the broadcasted shapes</returns>
        public static int[] GetBroadcastedShapes(int[] shapeArr1, int[] shapeArr2)
        {

            // Copy arrays to avoid reference issues 
            shapeArr1 = (int[])shapeArr1.Clone();
            shapeArr2 = (int[])shapeArr2.Clone();

            int maxShapeLength = (shapeArr1.Length >= shapeArr2.Length) ? shapeArr1.Length : shapeArr2.Length;
            int[] shapeRes = new int[maxShapeLength];

            int arr1Index = shapeArr1.Length - 1;
            int arr2Index = shapeArr2.Length - 1;


            int cntIndex = maxShapeLength - 1;
            while (cntIndex >= 0)
            {
                shapeRes[cntIndex] = (shapeArr1[arr1Index] >= shapeArr2[arr2Index]) ? shapeArr1[arr1Index] : shapeArr2[arr2Index];

                // decrement the indexes if possible
                if (arr1Index > 0)
                {
                    arr1Index--;
                }
                else
                {
                    // set this dim to 1, so the dim of the other array will always be >= to it
                    shapeArr1[arr1Index] = 1;
                }

                if (arr2Index > 0)
                {
                    arr2Index--;
                }
                else
                {
                    // set this dim to 1, so the dim of the other array will always be >= to it
                    shapeArr2[arr2Index] = 1;
                }

                cntIndex--;
            }

            return shapeRes;

        }

        /// <summary>
        /// Extend a 2D Array to broadcast an operation, for example, to perform (2,3) + (2,1), 
        /// this method will be used to extend (2,1) into (2,3)
        /// </summary>
        /// <param name="currentArray">The array that needs to be extended</param>
        /// <param name="newShape">The new shapes</param>
        /// <returns>The extended array</returns>
        public static NDimArray Extend2DArrayByShape(NDimArray currentArray, int[] newShape)
        {
            if (currentArray.Ndim > 2)
            {
                throw new NotImplementedException("Extending array is not yet supported for ndim>2");
            }

            if (currentArray.Shape.SequenceEqual(newShape))
            {
                return currentArray;
            }

            NDimArray res = new NDimArray(newShape);
            // Case where the array has only one dimension (multiple columns and one line)
            if (currentArray.Ndim == 1 || (currentArray.Ndim == 2 && currentArray.Shape[0] == 1))
            {
                int nbRowsToAdd = newShape[0];
                for (int i = 0; i < nbRowsToAdd; i++)
                {
                    // The number of columns is contained in shape[1] if there is two dimensions, and contained in shape[0] if there is one dimension
                    for (int j = 0; j < ((currentArray.Ndim == 2) ? currentArray.Shape[1] : currentArray.Shape[0]); j++)
                    {
                        if (currentArray.Ndim == 1)
                        {
                            res[i, j] = currentArray[j];
                        }
                        else
                        {
                            res[i, j] = currentArray[0, j];
                        }
                    }
                }
            }
            // Case where the array has 2 dimensions, one column and multiple rows
            else if (currentArray.Shape[1] == 1)
            {
                int nbColToAdd = newShape[1];
                for (int i = 0; i < currentArray.Shape[0]; i++)
                {
                    for (int j = 0; j < nbColToAdd; j++)
                    {
                        res[i, j] = currentArray[i, 0];
                    }
                }
            }

            return res;
        }
        
        /// <summary>
        /// Apply an operation between two 2Darrays, and extend the shapes of the arrays if needed
        /// </summary>
        /// <param name="operation"> The operation that takes 2 doubles and return a double</param>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns>the results of the operation(arr1,arr2) as a new ndim array</returns>
        private static NDimArray ApplyBroadcastOperationBetween2DArrays(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2)
        {
            int[] shapeRes = GetBroadcastedShapes(arr1.Shape, arr2.Shape);

            NDimArray newArr1 = Extend2DArrayByShape(arr1, shapeRes); //return an extended version of this array if needed
            NDimArray newArr2 = Extend2DArrayByShape(arr2, shapeRes);

            return ApplyOperationBetweenNDimArray(operation, newArr1, newArr2);
        }

        /// <summary>
        /// Apply an operation between 2 ndim array
        /// </summary>
        /// <param name="operation"> The operation that takes 2 doubles and return a double</param>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns></returns>
        public static NDimArray ApplyOperation(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2)
        {
            // If the shapes are equals

            if (arr1.Shape.SequenceEqual(arr2.Shape))
            {
                return ApplyOperationBetweenNDimArray(operation, arr1, arr2);
            }

            // If one of the array is a scalar
            else if ((arr1.Ndim == 1 && arr1.Shape[0] == 1) || (arr2.Ndim == 1 && arr2.Shape[0] == 1))
            {
                return ApplyOperationWithScalar(operation, arr1, arr2);
            }
            else if (IsShapesBroadcastable(arr1.Shape, arr2.Shape))
            {

                //TODO: implement NDIM broadcasting, only 2D broadcasting is supported yet
                if (arr1.Shape.Length <= 2 && arr2.Shape.Length <= 2)
                {

                    return ApplyBroadcastOperationBetween2DArrays(operation, arr1, arr2);
                }
                else
                {
                    throw new NotImplementedException("Broadcast operation for n>2 ndimarray is not supported yet.");
                }
            }
            else
            {
                //Uses Diagnostics.StackTrace to get the name of the parent method that called this one (the operation name).
                throw new InvalidOperationException($"Can't apply {(new System.Diagnostics.StackTrace()).GetFrame(1).GetMethod().Name} between those ndimarray, dimensions are " + string.Join(", ", arr1.Shape) + " and " + string.Join(", ", arr2.Shape) + " which is incompatible.");
            }
        }


        /// <summary>
        /// + operator that perform addition between 2 ndim array
        /// </summary>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator +(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> addition = (a, b) => a + b;

            return ApplyOperation(addition, arr1, arr2);
        }

        /// <summary>
        /// + operator that perform addition between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="arr2">The ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator +(double scalar, NDimArray arr2)
        {
            return new NDimArray(scalar) + arr2;
        }

        /// <summary>
        /// + operator that perform addition between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="arr1">The ndim array</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator +(NDimArray arr1, double scalar)
        {
            return arr1 + new NDimArray(scalar);
        }

        /// <summary>
        /// - operator that negate a ndimarray
        /// </summary>
        /// <param name="arr1"></param>
        /// <returns>The results of - arr1</returns>
        public static NDimArray operator -(NDimArray arr1)
        {
            NDimArray arr2 = new NDimArray(arr1.Shape);
            arr2.FillWithValue(-1);
            Func<double, double, double> neg = (a, b) => a * b;
            
            return ApplyOperation(neg, arr1, arr2);
        }

        
        /// <summary>
        /// - operator that perform substraction between 2 ndim array
        /// </summary>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator -(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> substract = (a, b) => a - b;

            return ApplyOperation(substract, arr1, arr2);
        }

        /// <summary>
        /// - operator that perform substraction between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="arr2">The ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator -(double scalar, NDimArray arr2)
        {
            return new NDimArray(scalar) - arr2;
        }

        /// <summary>
        /// - operator that perform substraction between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="arr1">The ndim array</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator -(NDimArray arr1, double scalar)
        {
            return arr1 - new NDimArray(scalar);
        }


        /// <summary>
        /// * operator that perform multiplication between 2 ndim array
        /// </summary>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator *(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> mul = (a, b) => a * b;

            return ApplyOperation(mul, arr1, arr2);
        }

        /// <summary>
        /// * operator that perform multiplication between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="arr2">The ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator *(double scalar, NDimArray arr2)
        {
            return new NDimArray(scalar) * arr2;
        }

        /// <summary>
        /// * operator that perform multiplication between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="arr1">The ndim array</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator *(NDimArray arr1, double scalar)
        {
            return arr1 * new NDimArray(scalar);
        }


        /// <summary>
        /// / operator that perform true division between 2 ndim array
        /// </summary>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator /(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> truediv = (a, b) => a / b;

            return ApplyOperation(truediv, arr1, arr2);
        }


        /// <summary>
        /// / operator that perform true division between a scalar (as a double) and a ndim array 
        /// </summary>
        /// <param name="scalar">The scalar</param>
        /// <param name="arr2">The ndim array</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator /(double scalar, NDimArray arr2)
        {
            return new NDimArray(scalar) / arr2;
        }

        /// <summary>
        /// / operator that perform true division between a ndim array and a scalar (as a double)
        /// </summary>
        /// <param name="arr1">The ndim array</param>
        /// <param name="scalar">The scalar</param>
        /// <returns>The result of this operation as a new ndim array</returns>
        public static NDimArray operator /(NDimArray arr1, double scalar)
        {
            return arr1 / new NDimArray(scalar);
        }


        /// <summary>
        /// Perform log base e operation on the values of this ndim array
        /// </summary>
        /// <param name="arr1">The array</param>
        /// <returns>The result of ln arr1</returns>
        public static NDimArray Log(NDimArray arr1)
        {
            NDimArray res = new NDimArray(arr1.Shape);
            for (int i = 0; i < arr1.NbElements; i++)
            {
                res.DataArray[i] = Math.Log(arr1.DataArray[i]);
            }
            return res;
        }

        /// <summary>
        /// Perform matrix multiplication between two ndim array
        /// </summary>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <returns>The result of arr1 @ arr2</returns>
        public static NDimArray Matmul(NDimArray arr1, NDimArray arr2)
        {
            //TODO:support ndim matmul (broadcasting)
            if (arr1.Ndim == 2 && arr2.Ndim == 2)
            {
                return MatmulBetween2DArrays(arr1, arr2, false, false);
            }
            else if (arr1.Ndim == 1 && arr2.Ndim == 2)
            {
                //Substitute arr1 by the same array but as a 2dim array with one row to allow matmul
                //arr1.Shape = new int[] { 1, arr1.Shape[0] };
                NDimArray arr1Reshaped = new NDimArray(new int[] { 1, arr1.Shape[0] }, arr1.DataArray);
                return MatmulBetween2DArrays(arr1Reshaped, arr2, true, false);
            }
            else if (arr1.Ndim == 2 && arr2.Ndim == 1)
            {
                //reshape arr2 as a 2dim array with one column to allow matmul
                //Substitute arr2 by the same array but as a 2dim array with one column to allow matmul
                //arr2.Shape = new int[] { arr2.Shape[0], 1 };
                NDimArray arr2Reshaped = new NDimArray(new int[] { arr2.Shape[0], 1 }, arr2.DataArray);
                return MatmulBetween2DArrays(arr1, arr2Reshaped, false, true);
            }
            else if (arr1.Ndim == 1 && arr2.Ndim == 1)
            {
                //reshape arr1 and arr2 to allow matmul
                //arr1.Shape = new int[] { 1, arr1.Shape[0] };
                //arr2.Shape = new int[] { arr2.Shape[0], 1 };
                NDimArray arr1Reshaped = new NDimArray(new int[] { 1, arr1.Shape[0] }, arr1.DataArray);
                NDimArray arr2Reshaped = new NDimArray(new int[] { arr2.Shape[0], 1 }, arr2.DataArray);
                //Substitute the 2 arrays by two 2Dim array with the same data to allow matmul
                return MatmulBetween2DArrays(arr1Reshaped, arr2Reshaped, true, true);
            }
            else
            {
                throw new NotImplementedException("Can't do matrix multiplication with other than 2d array, array1 is " + arr1.Ndim + " and array2 is " + arr2.Ndim + ".");
            }


        }

        

        /// <summary>
        /// Reshape a 2dim array into a 1dim array
        /// Only works for 2D array (used after for matmul)
        /// </summary>
        /// <param name="arr">The array to reshape</param>
        /// <param name="isArr1Extended">true if arr1 has been extended before</param>
        /// <param name="isArr2Extended">true if arr2 has been extended before</param>
        private static void ReshapeAs1DArray(NDimArray arr, bool isArr1Extended, bool isArr2Extended)
        {

            // If the two shapes are 1, it's a scalar
            if (arr.Shape[0] == 1 && arr.Shape[1] == 1)
            {
                arr.Shape = new int[] { 1 };
            }
            // If the array has been extended to be able to do matmul, remove the added dimension of the result
            else if (isArr1Extended || isArr2Extended)
            {
                if (arr.Shape[0] == 1)
                {
                    arr.Shape = new int[] { arr.Shape[1] };
                }
                else if (arr.Shape[1] == 1)
                {
                    arr.Shape = new int[] { arr.Shape[0] };
                }
            }

        }

        /// <summary>
        /// Perform matrix multiplication between two 2dim array
        /// </summary>
        /// <param name="arr1">The first ndim array</param>
        /// <param name="arr2">The second ndim array</param>
        /// <param name="isArr1Extended">True if arr1 has been extended</param>
        /// <param name="isArr2Extended">True if arr2 has been extended</param>
        /// <returns>Return the results of arr1 @ arr2</returns>
        private static NDimArray MatmulBetween2DArrays(NDimArray arr1, NDimArray arr2, bool isArr1Extended, bool isArr2Extended)
        {
            if (arr1.Shape[1] == arr2.Shape[0])
            {
                NDimArray res = new NDimArray(new int[] { arr1.Shape[0], arr2.Shape[1] });
                double val;
                int commonShapeIndex;
                int commonShape = arr1.Shape[1];
                for (int col = 0; col < res.Shape[0]; col++)
                {
                    for (int row = 0; row < res.Shape[1]; row++)
                    {
                        commonShapeIndex = 0;
                        val = 0;
                        while (commonShapeIndex < commonShape)
                        {
                            val += arr1[col, commonShapeIndex] * arr2[commonShapeIndex, row];
                            commonShapeIndex++;
                        }
                        res[col, row] = val;
                    }
                }
                //Reshape in 1D array if possible
                ReshapeAs1DArray(res, isArr1Extended, isArr2Extended);


                return res;
            }
            else
            {
                throw new InvalidOperationException("Can't do matrix multiplication between shapes : (" + string.Join(", ", arr1.Shape) + ") and (" + string.Join(", ", arr2.Shape) + "). (maybe an array has been automatically reshaped to allow matmul?");
            }
        }



    }


}

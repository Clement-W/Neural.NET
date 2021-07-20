//TODO: Generalize methods like XXX to support more than 2D arrays
using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    public class NDimArray
    {

        public double[] DataArray { get; private set; }

        // would be [5,3] for a (5x3) 2-darray 
        private int[] _shape;

        // list of indexes used to access the one dimensional array that
        // represents the n dimensional array
        public int[] StepIndexes { get; private set; }

        public int Ndim
        {
            get
            {
                return Shape.Length;
            }
        }


        public int[] Shape
        {
            get
            {
                return _shape;
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



        // return the number of elements contained in the ndim array
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
                        // If the data array is null, return the future data length by multitplying every shapes
                        return Shape.Aggregate(1, (a, b) => a * b);
                    }
                    else
                    {
                        return DataArray.Length;
                    }
                }
            }
        }

        public NDimArray(int[] shape)
        {
            Shape = shape;
            DataArray = new double[NbElements];
        }

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

        public NDimArray(NDimArray arr)
        {
            Shape = arr.Shape;
            DataArray = arr.DataArray;
        }

        // Used to create a 1DimArray
        public NDimArray(params double[] data)
        {
            Shape = new int[] { data.Length };
            DataArray = data;
        }

        public override string ToString()
        {
            return $"[{string.Join(", ", DataArray)}]";
        }


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




        public void FillWithValue(double value)
        {
            for (int i = 0; i < NbElements; i++)
            {
                DataArray[i] = value;
            }
        }

        // Convert a list of indexes like [2,3] to the corresponding index in the one-dim array that contains the data
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


        public static NDimArray Zeros_like(NDimArray arr)
        {
            return new NDimArray(arr.Shape);
        }

        public static NDimArray Ones_like(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            res.FillWithValue(1);
            return res;
        }

        // Sum the element of an array, or sum along the given axis
        //TODO: broadcasting is not supported for more than 2D arrays,
        // so this function doesn't support axis >=2 (only null, 0 or 1)
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
            else if (axis > Ndim - 1)
            {
                throw new InvalidOperationException($"Can't sum the array along axis {axis}, this array has only {Ndim} dimensions.");
            }
            else
            {
                throw new NotImplementedException($"Sum with a specified axis is not yet supported for ndim>2");
            }
        }



        public static NDimArray Tanh(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            for (int i = 0; i < arr.NbElements; i++)
            {
                res.DataArray[i] = Math.Tanh(arr.DataArray[i]);
            }
            return res;
        }

        public static NDimArray Exp(NDimArray arr)
        {
            NDimArray res = new NDimArray(arr.Shape);
            for (int i = 0; i < arr.NbElements; i++)
            {
                res.DataArray[i] = Math.Exp(arr.DataArray[i]);
            }
            return res;
        }

        public NDimArray Transpose()
        {
            //TODO: support n-dim transpose, not only 2dim
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


        //Two dimensions are compatible when they are equal, or one of them is 1
        // https://numpy.org/doc/stable/user/basics.broadcasting.html
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



        private static NDimArray ApplyOperationBetweenNDimArray(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2)
        {
            NDimArray res = new NDimArray(arr1.Shape);
            for (int i = 0; i < arr1.NbElements; i++)
            {
                res.DataArray[i] = operation(arr1.DataArray[i], arr2.DataArray[i]);
            }
            return res;
        }

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


        // Warning : Messy code here
        public static int[] GetBroadcastedShapes(int[] shapeArr1, int[] shapeArr2)
        {

            // Copy arrays to avoid reference issues 
            shapeArr1 = (int[])shapeArr1.Clone();
            shapeArr2 = (int[])shapeArr2.Clone();
            int maxShapeLength = (shapeArr1.Length >= shapeArr2.Length) ? shapeArr1.Length : shapeArr2.Length;

            int[] shapeRes = new int[maxShapeLength];

            int arr1Index = shapeArr1.Length - 1;
            int arr2Index = shapeArr2.Length - 1;

            int cptIndex = maxShapeLength - 1;
            while (cptIndex >= 0)
            {
                shapeRes[cptIndex] = (shapeArr1[arr1Index] >= shapeArr2[arr2Index]) ? shapeArr1[arr1Index] : shapeArr2[arr2Index];

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

                cptIndex--;
            }

            return shapeRes;

        }

        // Warning : Messy code here
        public static NDimArray Extend2DArrayByShape(NDimArray currentArray, int[] newShape)
        {
            if (currentArray.Shape.SequenceEqual(newShape))
            {
                return currentArray;
            }

            NDimArray res = new NDimArray(newShape);
            // Case where the array as only one dimension (multiple columns and one line)
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
            // Case where the array as 2 dimensions, one column and multiple rows
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

        private static NDimArray ApplyBroadcastOperationBetween2DArrays(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2)
        {

            int[] shapeRes = GetBroadcastedShapes(arr1.Shape, arr2.Shape);

            NDimArray newArr1 = Extend2DArrayByShape(arr1, shapeRes); //return an extended version of this array if needed
            NDimArray newArr2 = Extend2DArrayByShape(arr2, shapeRes);

            return ApplyOperationBetweenNDimArray(operation, newArr1, newArr2);
        }

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
                throw new InvalidOperationException("Can't apply this operation between those ndimarray, dimensions are " + string.Join(", ", arr1.Shape) + " and " + string.Join(", ", arr2.Shape) + " which is incompatible.");
            }
        }


        public static NDimArray operator +(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> addition = (a, b) => a + b;

            return ApplyOperation(addition, arr1, arr2);
        }

        public static NDimArray operator -(NDimArray arr1)
        {
            NDimArray arr2 = new NDimArray(arr1.Shape);
            arr2.FillWithValue(-1);
            Func<double, double, double> neg = (a, b) => a * b;

            return ApplyOperation(neg, arr1, arr2);
        }


        public static NDimArray operator -(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> substract = (a, b) => a - b;

            return ApplyOperation(substract, arr1, arr2);
        }

        public static NDimArray operator *(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> mul = (a, b) => a * b;

            return ApplyOperation(mul, arr1, arr2);
        }

        public static NDimArray operator /(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> truediv = (a, b) => a / b;

            return ApplyOperation(truediv, arr1, arr2);
        }

        public static NDimArray Matmul(NDimArray arr1, NDimArray arr2)
        {
            //TODO:support ndim matmul (broadcasting)
            if (arr1.Ndim == 2 && arr2.Ndim == 2)
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
                    return res;
                }
                else
                {
                    throw new InvalidOperationException("Can't do matrix multiplication between shapes : (" + string.Join(", ", arr1.Shape) + ") and (" + string.Join(", ", arr2.Shape) + ").");
                }
            }
            else
            {
                throw new NotImplementedException("Can't do matrix multiplication with other than 2d array, array1 is " + arr1.Ndim + " and array2 is " + arr2.Ndim + ".");
            }
        }

    }
}

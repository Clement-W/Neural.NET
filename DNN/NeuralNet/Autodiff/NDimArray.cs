using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    public class NDimArray
    {

        public double[] Data { get; private set; }

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
            private set { }
        }


        public int[] Shape
        {
            get
            {
                return _shape;
            }
            set
            {
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
                    return Shape.Aggregate(1, (a, b) => a * b);
                }
            }
        }

        public NDimArray(params int[] shape)
        {
            Shape = shape;
            Data = new double[NbElements];
        }

        public NDimArray(int[] shape, params double[] newData)
        {
            Shape = shape;
            if (NbElements == newData.Length)
            {
                Data = newData;
            }
            else
            {
                throw new ArgumentException("The given shape, and array are not compatible.");
            }

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
                Data[i] = value;
            }
        }


        private int getIndexInDataArray(int[] indexes)
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
                return Data[getIndexInDataArray(indexes)];
            }
            set
            {
                Data[getIndexInDataArray(indexes)] = value;
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


        //Two dimensions are compatible when they are equal, or one of them is 1
        // https://numpy.org/doc/stable/user/basics.broadcasting.html
        public static bool IsOperationBroadcastable(int[] shape1, int[] shape2)
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
                res.Data[i] = operation(arr1.Data[i], arr2.Data[i]);
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
                    res.Data[i] = operation(arr1.Data[i], arr2.Data[0]);
                }
                else
                {
                    res.Data[i] = operation(arr1.Data[0], arr2.Data[i]);
                }
            }
            return res;
        }

        public static NDimArray ApplyOperation(Func<double, double, double> operation, NDimArray arr1, NDimArray arr2){
            if (arr1.Shape.SequenceEqual(arr2.Shape))
            {
                return ApplyOperationBetweenNDimArray(operation, arr1, arr2);
            }

            else if ((arr1.Ndim == 1 && arr1.Shape[0] == 1) || (arr2.Ndim == 1 && arr2.Shape[0] == 1))
            {
                return ApplyOperationWithScalar(operation, arr1, arr2);
            }
            else if (IsOperationBroadcastable(arr1.Shape, arr2.Shape))
            {
                //TODO: implement broadcasting
                throw new NotImplementedException("Broadcast operation is not supported yet.");
            }
            else
            {
                throw new InvalidOperationException("Can't apply this operation between those ndimarray, dimensions are " + arr1.Shape + " and " + arr2.Shape + " which is incompatible.");
            }
        }


        public static NDimArray operator +(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> addition = (a, b) => a + b;

            return ApplyOperation(addition,arr1,arr2);
        }


        public static NDimArray operator -(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> substract = (a, b) => a - b;

            return ApplyOperation(substract,arr1,arr2);
        }

        public static NDimArray operator *(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> mul = (a, b) => a * b;

            return ApplyOperation(mul,arr1,arr2);
        }

        public static NDimArray operator /(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> truediv = (a, b) => a / b;

            return ApplyOperation(truediv,arr1,arr2);
        }



        



    }
}

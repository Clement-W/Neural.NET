using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    public class NDimArray
    {

        public double[] Data{get;private set;}

        // would be [5,3] for a (5x3) 2-darray 
        private int[] _shape;

        // list of indexes used to access the one dimensional array that
        // represents the n dimensional array
        public int[] StepIndexes { get; private set; }


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
            int[] steps = new int[Shape.Length];

            int step = 1;
            for (int i = Shape.Length - 1; i >= 0; i--)
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
            if (indexes.Length == Shape.Length)
            {
                int index = 0;
                for (int i = 0; i < Shape.Length; i++)
                {
                    index += indexes[i] * StepIndexes[i];
                }
                return index;
            }
            else
            {
                throw new ArgumentException("The number of indexes given (" + indexes.Length + ")" + " doesn't fit the array shape (" + Shape.Length + ").");
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
            if (Shape.Length == 2)
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
            for(int i = minShapeLength-1; i>=0;i--){
                if(shape1[i] != 1 && shape2[i] != 1 && shape1[i] != shape2[i]){
                    return false;
                }
            }
            return true;
        }



        private static NDimArray ApplyOperationBetweenNDimArray(Func<double,double,double> operation, NDimArray arr1, NDimArray arr2){
            NDimArray res = new NDimArray(arr1.Shape);
            for (int i = 0; i < arr1.NbElements; i++)
            {
                res.Data[i] = operation(arr1.Data[i],arr2.Data[i]);
            }
            return res;
        }
        
        private static NDimArray ApplyOperationWithScalar(Func<double,double,double> operation, NDimArray arr, NDimArray scalarArr){
            
            NDimArray res = new NDimArray(arr.Shape);
            for (int i = 0; i < arr.NbElements; i++)
            {
                res.Data[i] = operation(arr.Data[i],scalarArr.Data[0]);
            }
            return res;
        }
        
        
        public static NDimArray operator +(NDimArray arr1, NDimArray arr2)
        {
            Func<double,double,double> addition = (a,b) => a + b;

            if (arr1.Shape.SequenceEqual(arr2.Shape))
            {
                return ApplyOperationBetweenNDimArray(addition,arr1,arr2);
            }
            
            else if(arr1.Shape.Length==1 && arr1.Shape[0]==1){
                return ApplyOperationWithScalar(addition,arr2,arr1);
            }
            else if(arr2.Shape.Length==1 && arr2.Shape[0]==1){
                return ApplyOperationWithScalar(addition,arr1,arr2);
            }

            else if(IsOperationBroadcastable(arr1.Shape,arr2.Shape)){
                int[] newShape = new int[]{};
                //TODO:
                
            }
            return null;
        }




    }
}

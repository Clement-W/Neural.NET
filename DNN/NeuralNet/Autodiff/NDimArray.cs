using System;
using System.Linq;

namespace NeuralNet.Autodiff
{
    public class NDimArray
    {

        private double[] _data;

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
            _data = new double[NbElements];
        }

        public NDimArray(int[] shape, params double[] newData)
        {
            Shape = shape;
            if (NbElements == newData.Length)
            {
                _data = newData;
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
                _data[i] = value;
            }
        }


        private int getIndexIn1DArray(int[] indexes)
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
                return _data[getIndexIn1DArray(indexes)];
            }
            set
            {
                _data[getIndexIn1DArray(indexes)] = value;
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


        private bool IsOperationBroadcastable(){
            return false;
        }

        public static NDimArray operator +(NDimArray arr1, NDimArray arr2){
            
            if(arr1.Shape == arr2.Shape){
                NDimArray res = new NDimArray(arr1.Shape);
                for(int i =0;i<arr1.NbElements;i++){
                    res[i] = arr1[i] + arr2[i];
                }
                return res;
            }

            //else if(arr1.)
            return null;
        }




    }
}
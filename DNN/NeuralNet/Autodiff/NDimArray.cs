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
                    if(Data == null){
                        // If the data array is null, return the future data length by multitplying every shapes
                        return Shape.Aggregate(1, (a, b) => a * b);
                    }else{
                        return Data.Length;
                    }      
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

        public NDimArray(NDimArray arr)
        {
            Shape = arr.Shape;
            Data = arr.Data;
        }

        public static NDimArray CreateScalar(double val)
        {
            return new NDimArray(new int[] { 1 }, val);
        }

        public override string ToString()
        {
            return $"[{string.Join(", ", Data)}]";
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
                return Data[ConvertNDIndexTo1DIndex(indexes)];
            }
            set
            {
                Data[ConvertNDIndexTo1DIndex(indexes)] = value;
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


        public static NDimArray Zeros_like(NDimArray arr){
            return new NDimArray(arr.Shape);
        }

        public static NDimArray Ones_like(NDimArray arr){
            NDimArray res = new NDimArray(arr.Shape);
            res.FillWithValue(1);
            return res;
        }

        public double Sum(){
            return this.Data.Sum();
        }



        public static NDimArray Tanh(NDimArray arr){
            NDimArray res = new NDimArray(arr.Shape);
            for(int i =0;i<arr.NbElements;i++){
                res.Data[i] = Math.Tanh(arr.Data[i]);
            }
            return res;
        }

        public static NDimArray Exp(NDimArray arr){
            NDimArray res = new NDimArray(arr.Shape);
            for(int i =0;i<arr.NbElements;i++){
                res.Data[i] = Math.Exp(arr.Data[i]);
            }
            return res;
        }

        public NDimArray Transpose(){
            //TODO: support n-dim transpose, not only 2dim
            if(Ndim==1){
                return new NDimArray(this);
            }else if(Ndim==2){
                NDimArray res = new NDimArray(new int[]{Shape[1],Shape[0]});
                for(int newCol =0;newCol<res.Shape[0];newCol++){
                    for(int newRow =0;newRow<res.Shape[1];newRow++){
                        res[newCol,newRow] = this[newRow,newCol];
                    }
                }
                return res;
            }else{
                throw new NotImplementedException("Can't transpose a ndimarray with n > 2 (TODO)");
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


        public static int[] GetBroadcastedShapes(int[] shapeArr1,int[] shapeArr2)
        {
            int maxShapeLength = (shapeArr1.Length >= shapeArr2.Length) ? shapeArr1.Length : shapeArr2.Length;

            int[] shapeRes = new int[maxShapeLength];

            int arr1Index = shapeArr1.Length-1;
            int arr2Index = shapeArr2.Length-1;

            int cptIndex = maxShapeLength-1;
            while(cptIndex >= 0)
            {
                shapeRes[cptIndex] = (shapeArr1[arr1Index] >= shapeArr2[arr2Index]) ? shapeArr1[arr1Index] : shapeArr2[arr2Index];

                // decrement the indexes if possible
                if (arr1Index > 0) {
                    arr1Index--;
                }
                else
                {
                    shapeArr1[arr1Index] = 1; // set this dim to 1, so the dim of the other array will always be >= 
                }

                if (arr2Index > 0) {
                    arr2Index--;
                }
                else
                {
                    shapeArr2[arr2Index] = 1;
                }

                cptIndex--;
            }

            return shapeRes;
            
        }

        public static NDimArray Extend2DArrayByShape(NDimArray currentArray, int[] newShape)
        {
            if (currentArray.Shape.SequenceEqual(newShape))
            {
                return currentArray;
            }

            NDimArray res = new NDimArray(newShape);
            // Case where the array as only one dimension (multiple columns and one line)
            if (currentArray.Ndim == 1 || (currentArray.Ndim == 2 && currentArray.Shape[0]==1))
            {
                int nbRowsToAdd = newShape[0];
                for(int i = 0; i < nbRowsToAdd; i++)
                {
                    // The number of columns is contained in shape[1] if there is two dimensions, and contained in shape[0] if there is one dimension
                    for(int j = 0; j < ((currentArray.Ndim == 2) ? currentArray.Shape[1] : currentArray.Shape[0]); j++)
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
            else if (IsOperationBroadcastable(arr1.Shape, arr2.Shape))
            {
                //TODO: implement NDIM broadcasting, only 2D broadcasting is supported yet
                if(arr1.Shape.Length <=2 && arr2.Shape.Length <= 2)
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
                throw new InvalidOperationException("Can't apply this operation between those ndimarray, dimensions are " + arr1.Shape + " and " + arr2.Shape + " which is incompatible.");
            }
        }


        public static NDimArray operator +(NDimArray arr1, NDimArray arr2)
        {
            Func<double, double, double> addition = (a, b) => a + b;

            return ApplyOperation(addition, arr1, arr2);
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
                    throw new InvalidOperationException("Can't do matrix multiplication between shapes : (" + string.Join(", " ,arr1.Shape) + ") and (" + string.Join(", ",arr2.Shape) + ").");
                }
            }else{
                throw new NotImplementedException("Can't do matrix multiplication with other than 2d array, array1 is " + arr1.Ndim + " and array2 is " + arr2.Ndim  + ".");
            }
        }

    }
}

using Xunit;
using NeuralNet.Autodiff;
using System;
using System.Linq;

namespace NeuralNet.UnitTests
{
    public class NeuralNet_NDimArray_Tests
    {

        private NDimArray arr1;
        private NDimArray arr2;
        private NDimArray arr3;

        public NeuralNet_NDimArray_Tests()
        {
            // new object created for each method
            arr1 = new NDimArray(3, 4);
            arr2 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            arr3 = new NDimArray(1);

        }

        [Fact]
        public void NDimArray_Test_Shape()
        {
            //Console.WriteLine("[{0}]", string.Join(", ", arr.Shape));
            //Console.WriteLine(arr.Shape.GetType());
            Assert.True(arr1.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(arr2.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(arr3.Shape.SequenceEqual(new int[] { 1 }));
        }

        [Fact]
        public void NDimArray_Test_NbElements()
        {
            Assert.True(arr1.NbElements == 12);
            Assert.True(arr2.NbElements == 12);
            Assert.True(arr3.NbElements == 1);
        }

        [Fact]
        public void NDimArray_Test_StepIndexes()
        {
            Assert.True(arr1.StepIndexes.SequenceEqual(new int[] { 4, 1 }));
            Assert.True(arr2.StepIndexes.SequenceEqual(new int[] { 6, 2, 1 }));
            Assert.True(arr3.StepIndexes.SequenceEqual(new int[] { 1 }));
        }

        [Fact]
        public void NDimArray_Test_ReadDefaultValue()
        {
            Assert.True(arr1[1, 1] == 0);
            Assert.True(arr2[1, 1, 1] == 10);
            Assert.True(arr3[0] == 0);
        }

        [Fact]
        public void NDimArray_Test_FillWithValue()
        {
            arr1.FillWithValue(2);
            Assert.True(arr1[1, 1] == 2);
            arr2.FillWithValue(-29.9);
            Assert.True(arr2[1, 1, 1] == -29.9);
            arr3.FillWithValue(100);
            Assert.True(arr3[0] == 100);

        }

        [Fact]
        public void NDimArray_Test_SetValue()
        {
            arr1[2, 1] = 10;
            Assert.True(arr1[2, 1] == 10);
            arr2[1, 2, 0] = 10;
            Assert.True(arr2[1, 2, 0] == 10);
            arr3[0] = 10;
            Assert.True(arr3[0] == 10);
        }

        [Fact]
        public void NDimArray_Test_PrintAsMatrix()
        {
            arr1.FillWithValue(5);
            arr1[1, 1] = 0;
            //arr1.PrintAsMatrix();

            arr2.FillWithValue(1.129);
            Assert.Throws<InvalidOperationException>(() => arr2.PrintAsMatrix());
            Assert.Throws<InvalidOperationException>(() => arr3.PrintAsMatrix());
        }

        [Fact]
        public void NDimArray_Test_Broadcastable()
        {
            NDimArray a = new NDimArray(new int[] { 1 });
            NDimArray b = new NDimArray(new int[] { 1, 1 });
            NDimArray c = new NDimArray(new int[] { 3, 2 });
            NDimArray e = new NDimArray(new int[] { 1, 2 });
            NDimArray f = new NDimArray(new int[] { 1, 1, 2 });
            NDimArray g = new NDimArray(new int[] { 1, 1, 1, 1, 1, 1 });
            NDimArray h = new NDimArray(new int[] { 3, 5 });
            NDimArray i = new NDimArray(new int[] { 2 });
            Assert.True(NDimArray.IsOperationBroadcastable(a.Shape, b.Shape));
            Assert.True(NDimArray.IsOperationBroadcastable(a.Shape, c.Shape));
            Assert.True(NDimArray.IsOperationBroadcastable(c.Shape, e.Shape));
            Assert.True(NDimArray.IsOperationBroadcastable(c.Shape, f.Shape));
            Assert.True(NDimArray.IsOperationBroadcastable(e.Shape, f.Shape));
            Assert.True(NDimArray.IsOperationBroadcastable(a.Shape, g.Shape));
            Assert.False(NDimArray.IsOperationBroadcastable(c.Shape, h.Shape));
            Assert.True(NDimArray.IsOperationBroadcastable(c.Shape, i.Shape));
            Assert.True(h.Shape.SequenceEqual(new int[] { 3, 5 }));
        }

        [Fact]
        public void NDimArray_Test_Addition_Array_With_Scalar()
        {
            arr1.FillWithValue(1);
            arr3.FillWithValue(2);
            NDimArray a = arr1 + arr3;
            NDimArray b = arr3 + arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(a.Data.SequenceEqual(new double[] { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 }));
            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.Data.SequenceEqual(new double[] { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 }));
        }

        [Fact]
        public void NDimArray_Test_Addition_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(1);

            NDimArray a = a1 + a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.Data.SequenceEqual(new double[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }));
        }

        [Fact]
        public void NDimArray_Test_Addition_Incompatible()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 5, 4, 3 });
            NDimArray a3 = new NDimArray(new int[] { 3, 2 });
            a2.FillWithValue(1);

            Assert.Throws<InvalidOperationException>(() => a1 + a2);
            Assert.Throws<NotImplementedException>(() => a1 + a3);

        }

        /*
        [Fact]
        public void NDimArray_Test_Broadcast_Addition_Operation()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30);
            NDimArray a2 = new NDimArray(new int[] { 3 }, 0, 1, 2);
            NDimArray a3 = new NDimArray(new int[] { 4, 1 }, 0, 10, 20, 30);
            //NDimArray a4 = new NDimArray(new int[] { 1, 1, 1, 1, 4, 1 }, 0, 10, 20, 30);

            NDimArray a4 = a1 + a2;
            NDimArray a5 = a2 + a3;
            Assert.True(a4.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a5.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a4.Data.SequenceEqual(new double[] { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 }));
            Assert.True(a5.Data.SequenceEqual(new double[] { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 }));
        }*/


        [Fact]
        public void NDimArray_Test_Substract_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(1);

            NDimArray a = a1 - a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.Data.SequenceEqual(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }));
        }


        [Fact]
        public void NDimArray_Test_Substract_Array_With_Scalar()
        {
            arr1.FillWithValue(1);
            arr3.FillWithValue(2);
            NDimArray a = arr1 - arr3;
            NDimArray b = arr3 - arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(a.Data.SequenceEqual(new double[] { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }));
            Assert.True(arr1.Data.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
            Assert.True(arr3.Data.SequenceEqual(new double[] { 2 }));
            
            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.Data.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
        }

        [Fact]
        public void NDimArray_Test_Substract_Incompatible()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 5, 4, 3 });
            NDimArray a3 = new NDimArray(new int[] { 3, 2 });
            a2.FillWithValue(1);

            Assert.Throws<InvalidOperationException>(() => a1 - a2);
            Assert.Throws<NotImplementedException>(() => a1 - a3);

        }


        [Fact]
        public void NDimArray_Test_Mul_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(2);

            NDimArray a = a1 * a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.Data.SequenceEqual(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 }));
        }


        [Fact]
        public void NDimArray_Test_Mul_Array_With_Scalar()
        {
            arr1.FillWithValue(3);
            arr3.FillWithValue(2);
            NDimArray a = arr1 * arr3;
            NDimArray b = arr3 * arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(a.Data.SequenceEqual(new double[] { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 }));
            
            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.Data.SequenceEqual(a.Data));
        }

        [Fact]
        public void NDimArray_Test_Mul_Incompatible()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 5, 4, 3 });
            NDimArray a3 = new NDimArray(new int[] { 3, 2 });
            a2.FillWithValue(1);

            Assert.Throws<InvalidOperationException>(() => a1 * a2);
            Assert.Throws<NotImplementedException>(() => a1 * a3);

        }




        [Fact]
        public void NDimArray_Test_Truediv_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(2);

            NDimArray a = a1 / a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.Data.SequenceEqual(new double[] { 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6 }));
        }


        [Fact]
        public void NDimArray_Test_Truediv_Array_With_Scalar()
        {
            arr1.FillWithValue(3);
            arr3.FillWithValue(6);
            arr1[0,1]=6;
            NDimArray a = arr1 / arr3;
            NDimArray b = arr3 / arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));
            Console.WriteLine("[{0}]",string.Join(", ",a.Data));
            Assert.True(a.Data.SequenceEqual(new double[] { 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }));
            
            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.Data.SequenceEqual(new double[] { 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }));
        }

        [Fact]
        public void NDimArray_Test_Truediv_Incompatible()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 5, 4, 3 });
            NDimArray a3 = new NDimArray(new int[] { 3, 2 });
            a2.FillWithValue(1);

            Assert.Throws<InvalidOperationException>(() => a1 / a2);
            Assert.Throws<NotImplementedException>(() => a1 / a3);

        }





    }
}

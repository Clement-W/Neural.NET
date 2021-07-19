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

        private NDimArray arr4;

        public NeuralNet_NDimArray_Tests()
        {
            // new object created for each method
            arr1 = new NDimArray(new int[]{3, 4});
            arr2 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

            arr3 = new NDimArray(2);
            arr4 = new NDimArray(1,2,3,4,5,6);

        }

        [Fact]
        public void NDimArray_Test_Shape()
        {
            //Console.WriteLine("[{0}]", string.Join(", ", arr.Shape));
            //Console.WriteLine(arr.Shape.GetType());
            Assert.True(arr1.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(arr2.Shape.SequenceEqual(new int[] { 2, 3, 2 }));

            Assert.True(arr3.Shape.SequenceEqual(new int[] { 1 }));
            Assert.True(arr4.Shape.SequenceEqual(new int[] { 6 }));
            NDimArray a = new NDimArray(arr2);
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));

        }

        [Fact]
        public void NDimArray_Test_NbElements()
        {
            Assert.True(arr1.NbElements == 12);
            Assert.True(arr2.NbElements == 12);

            Assert.True(arr3.NbElements == 1);
            Assert.True(arr4.NbElements == 6);

        }

        [Fact]
        public void NDimArray_Test_StepIndexes()
        {
            Assert.True(arr1.StepIndexes.SequenceEqual(new int[] { 4, 1 }));
            Assert.True(arr2.StepIndexes.SequenceEqual(new int[] { 6, 2, 1 }));
            
            Assert.True(arr3.StepIndexes.SequenceEqual(new int[] { 1 }));
            Assert.True(arr4.StepIndexes.SequenceEqual(new int[] { 1 }));
        }

        [Fact]
        public void NDimArray_Test_ReadValue()
        {
            Assert.True(arr1[1, 1] == 0);
            Assert.True(arr2[1, 1, 1] == 10);

            Assert.True(arr3[0] == 2);

            Assert.True(arr4[0] == 1);
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
            arr4.FillWithValue(100);
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
            arr4[0] = 10;
            Assert.True(arr4[0] == 10);
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
        public void NDimArray_Test_IsShapesBroadcastable()
        {
            NDimArray a = new NDimArray(new int[] { 1 });
            NDimArray b = new NDimArray(new int[] { 1, 1 });
            NDimArray c = new NDimArray(new int[] { 3, 2 });
            NDimArray e = new NDimArray(new int[] { 1, 2 });
            NDimArray f = new NDimArray(new int[] { 1, 1, 2 });
            NDimArray g = new NDimArray(new int[] { 1, 1, 1, 1, 1, 1 });
            NDimArray h = new NDimArray(new int[] { 3, 5 });
            NDimArray i = new NDimArray(new int[] { 2 });
            Assert.True(NDimArray.IsShapesBroadcastable(a.Shape, b.Shape));
            Assert.True(NDimArray.IsShapesBroadcastable(a.Shape, c.Shape));
            Assert.True(NDimArray.IsShapesBroadcastable(c.Shape, e.Shape));
            Assert.True(NDimArray.IsShapesBroadcastable(c.Shape, f.Shape));
            Assert.True(NDimArray.IsShapesBroadcastable(e.Shape, f.Shape));
            Assert.True(NDimArray.IsShapesBroadcastable(a.Shape, g.Shape));
            Assert.False(NDimArray.IsShapesBroadcastable(c.Shape, h.Shape));
            Assert.True(NDimArray.IsShapesBroadcastable(c.Shape, i.Shape));
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
            Assert.True(a.DataArray.SequenceEqual(new double[] { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 }));
            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.DataArray.SequenceEqual(new double[] { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 }));
        }

        [Fact]
        public void NDimArray_Test_Addition_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(1);

            NDimArray a = a1 + a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.DataArray.SequenceEqual(new double[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }));
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

        [Fact]
        public void NDimArray_Test_Broadcast2D_Addition_Operation()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30);
            NDimArray a2 = new NDimArray(new int[] { 3 }, 0, 1, 2);
            NDimArray a3 = new NDimArray(new int[] { 1, 3 }, 0, 1, 2);
            NDimArray a4 = new NDimArray(new int[] { 4, 1 }, 0, 10, 20, 30);

            Assert.True(a1.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a2.Shape.SequenceEqual(new int[] { 3 }));
            Assert.True(a3.Shape.SequenceEqual(new int[] { 1, 3 }));

            NDimArray a5 = a1 + a2;
            NDimArray a6 = a1 + a3;

            NDimArray a7 = a1 + a4;
            NDimArray a8 = a3 + a4;
            NDimArray a9 = a2 + a4;

            Assert.True(a5.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a6.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a7.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a8.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a9.Shape.SequenceEqual(new int[] { 4, 3 }));

            Assert.True(a5.DataArray.SequenceEqual(new double[] { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 }));
            Assert.True(a6.DataArray.SequenceEqual(new double[] { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 }));
            Assert.True(a7.DataArray.SequenceEqual(new double[] { 0, 0, 0, 20, 20, 20, 40, 40, 40, 60, 60, 60 }));
            Assert.True(a8.DataArray.SequenceEqual(new double[] { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 }));
            Assert.True(a9.DataArray.SequenceEqual(new double[] { 0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32 }));

        }


        [Fact]
        public void NDimArray_Test_Substract_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(1);

            NDimArray a = a1 - a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.DataArray.SequenceEqual(new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }));
        }


        [Fact]
        public void NDimArray_Test_Substract_Array_With_Scalar()
        {
            arr1.FillWithValue(1);
            arr3.FillWithValue(2);
            NDimArray a = arr1 - arr3;
            NDimArray b = arr3 - arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(a.DataArray.SequenceEqual(new double[] { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }));
            Assert.True(arr1.DataArray.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
            Assert.True(arr3.DataArray.SequenceEqual(new double[] { 2 }));

            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.DataArray.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
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
        public void NDimArray_Test_Broadcast2D_Substract_Operation()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30);
            NDimArray a2 = new NDimArray(new int[] { 3 }, 0, 1, 2);
            NDimArray a3 = new NDimArray(new int[] { 1, 3 }, 0, 1, 2);
            NDimArray a4 = new NDimArray(new int[] { 4, 1 }, 0, 10, 20, 30);

            Assert.True(a1.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a2.Shape.SequenceEqual(new int[] { 3 }));
            Assert.True(a3.Shape.SequenceEqual(new int[] { 1, 3 }));

            NDimArray a5 = a1 - a2;
            NDimArray a6 = a1 - a3;
            NDimArray a7 = a1 - a4;
            NDimArray a8 = a4 - a3;
            NDimArray a9 = a4 - a2;

            Assert.True(a5.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a6.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a7.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a8.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a9.Shape.SequenceEqual(new int[] { 4, 3 }));

            Assert.True(a5.DataArray.SequenceEqual(new double[] { 0, -1, -2, 10, 9, 8, 20, 19, 18, 30, 29, 28 }));
            Assert.True(a6.DataArray.SequenceEqual(new double[] { 0, -1, -2, 10, 9, 8, 20, 19, 18, 30, 29, 28 }));
            Assert.True(a7.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
            Assert.True(a8.DataArray.SequenceEqual(new double[] { 0, -1, -2, 10, 9, 8, 20, 19, 18, 30, 29, 28 }));
            Assert.True(a9.DataArray.SequenceEqual(new double[] { 0, -1, -2, 10, 9, 8, 20, 19, 18, 30, 29, 28 }));

        }


        [Fact]
        public void NDimArray_Test_Mul_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(2);

            NDimArray a = a1 * a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.DataArray.SequenceEqual(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 }));
        }



        [Fact]
        public void NDimArray_Test_Mul_Array_With_Scalar2()
        {
            arr1.FillWithValue(3);
            arr3.FillWithValue(2);
            NDimArray a = arr1 * arr3;
            NDimArray b = arr3 * arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(a.DataArray.SequenceEqual(new double[] { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 }));

            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.DataArray.SequenceEqual(new double[] { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 }));
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
        public void NDimArray_Test_Broadcast2D_Mul_Operation()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30);
            NDimArray a2 = new NDimArray(new int[] { 3 }, 0, 1, 2);
            NDimArray a3 = new NDimArray(new int[] { 1, 3 }, 0, 1, 2);
            NDimArray a4 = new NDimArray(new int[] { 4, 1 }, 0, 10, 20, 30);

            Assert.True(a1.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a2.Shape.SequenceEqual(new int[] { 3 }));
            Assert.True(a3.Shape.SequenceEqual(new int[] { 1, 3 }));

            NDimArray a5 = a1 * a2;
            NDimArray a6 = a1 * a3;
            NDimArray a7 = a1 * a4;
            NDimArray a8 = a4 * a3;
            NDimArray a9 = a4 * a2;


            Assert.True(a5.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a6.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a7.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a8.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a9.Shape.SequenceEqual(new int[] { 4, 3 }));

            Assert.True(a5.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 10, 20, 0, 20, 40, 0, 30, 60 }));
            Assert.True(a6.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 10, 20, 0, 20, 40, 0, 30, 60 }));
            Assert.True(a7.DataArray.SequenceEqual(new double[] { 0, 0, 0, 100, 100, 100, 400, 400, 400, 900, 900, 900 }));
            Assert.True(a8.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 10, 20, 0, 20, 40, 0, 30, 60 }));
            Assert.True(a9.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 10, 20, 0, 20, 40, 0, 30, 60 }));

        }




        [Fact]
        public void NDimArray_Test_Truediv_Between_Arrays_Same_Shape()
        {
            NDimArray a1 = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 3, 2 });
            a2.FillWithValue(2);

            NDimArray a = a1 / a2;
            Assert.True(a.Shape.SequenceEqual(new int[] { 2, 3, 2 }));
            Assert.True(a.DataArray.SequenceEqual(new double[] { 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6 }));
        }


        [Fact]
        public void NDimArray_Test_Truediv_Array_With_Scalar()
        {
            arr1.FillWithValue(3);
            arr3.FillWithValue(6);
            arr1[0, 1] = 6;
            NDimArray a = arr1 / arr3;
            NDimArray b = arr3 / arr1;
            Assert.True(a.Shape.SequenceEqual(new int[] { 3, 4 }));

            Assert.True(a.DataArray.SequenceEqual(new double[] { 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }));

            Assert.True(b.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(b.DataArray.SequenceEqual(new double[] { 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }));
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

        [Fact]
        public void NDimArray_Test_Broadcast2D_TrueDiv_Operation()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24);
            NDimArray a2 = new NDimArray(new int[] { 3 }, 2, 2, 2);
            NDimArray a3 = new NDimArray(new int[] { 1, 3 }, 2, 2, 2);
            NDimArray a4 = new NDimArray(new int[] { 4, 1 }, 2, 2, 2, 2);

            Assert.True(a1.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a2.Shape.SequenceEqual(new int[] { 3 }));
            Assert.True(a3.Shape.SequenceEqual(new int[] { 1, 3 }));

            NDimArray a5 = a1 / a2;
            NDimArray a6 = a1 / a3;
            NDimArray a7 = a1 / a4;
            NDimArray a8 = a4 / a3;
            NDimArray a9 = a4 / a2;


            Assert.True(a5.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a6.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a7.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a8.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a9.Shape.SequenceEqual(new int[] { 4, 3 }));

            Assert.True(a5.DataArray.SequenceEqual(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
            Assert.True(a6.DataArray.SequenceEqual(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
            Assert.True(a7.DataArray.SequenceEqual(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
            Assert.True(a8.DataArray.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
            Assert.True(a9.DataArray.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));

        }


        [Fact]
        public void NDimArray_Test_Simple_Matmul()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 3, 2 }, 2, 2, 2, 2, 2, 2);

            NDimArray res = NDimArray.Matmul(a1, a2);

            Assert.True(res.Shape.SequenceEqual(new int[] { 4, 2 }));
            Assert.True(res.DataArray.SequenceEqual(new double[] { 12, 12, 30, 30, 48, 48, 66, 66 }));

        }

        [Fact]
        public void NDimArray_Test_Matmul_2DArray_Exception()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 1, 3, 2 }, 2, 2, 2, 2, 2, 2);

            Assert.Throws<NotImplementedException>(() => NDimArray.Matmul(a1, a2));

        }

        [Fact]
        public void NDimArray_Test_Matmul_Shape_Incompatible()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray a2 = new NDimArray(new int[] { 2, 4 }, 2, 2, 2, 2, 2, 2, 2, 2);

            Assert.Throws<InvalidOperationException>(() => NDimArray.Matmul(a1, a2));

        }


        [Fact]
        public void NDimArray_Test_Zeros_like()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray res = NDimArray.Zeros_like(a1);

            Assert.True(res.Shape.SequenceEqual(a1.Shape));
            Assert.True(res.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));

        }

        [Fact]
        public void NDimArray_Test_Ones_like()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray res = NDimArray.Ones_like(a1);

            Assert.True(res.Shape.SequenceEqual(a1.Shape));
            Assert.True(res.DataArray.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));

        }


        [Fact]
        public void NDimArray_Test_Sum_SimpleCase()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray res1 = a1.Sum();
            NDimArray res2 = a1.Sum(keepDims:true);

            Assert.True(res1.DataArray[0] == 78);
            Assert.True(res1.Shape.SequenceEqual(new int []{1}));

            Assert.True(res2.DataArray[0] == 78);
            Assert.True(res2.Shape.SequenceEqual(new int []{1,1}));

        }

        [Fact]
        public void NDimArray_Test_Sum_With_Axes()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray res2 = a1.Sum(axis: 0);
            NDimArray res3 = a1.Sum(axis: 1);
            NDimArray res4 = a1.Sum(axis: 1,keepDims:true);
            NDimArray res5 = a1.Sum(axis: 0,keepDims:true);

            Assert.True(res2.Shape.SequenceEqual(new int[] { 3 }));
            Assert.True(res2.DataArray.SequenceEqual(new double[] { 22,26,30}));

            Assert.True(res3.Shape.SequenceEqual(new int[] { 4 }));
            Assert.True(res3.DataArray.SequenceEqual(new double[] { 6,15,24,33}));

            Assert.True(res4.Shape.SequenceEqual(new int[] { 4,1 }));
            Assert.True(res4.DataArray.SequenceEqual(new double[] { 6,15,24,33}));

            Assert.True(res5.Shape.SequenceEqual(new int[] {1, 3 }));
            Assert.True(res5.DataArray.SequenceEqual(new double[] { 22,26,30}));

            Assert.Throws<InvalidOperationException>(() => a1.Sum(axis: 2));

        }

        [Fact]
        public void NDimArray_Test_Sum_Limit_Axes1()
        {

            NDimArray a1 = new NDimArray(new int[]{4,1},1,2,3,4);

            NDimArray res1 = a1.Sum();
            NDimArray res2 = a1.Sum(0);
            NDimArray res3 = a1.Sum(1);

            NDimArray res4 = a1.Sum(keepDims:true);
            NDimArray res5 = a1.Sum(0,true);
            NDimArray res6 = a1.Sum(1,true);

            Assert.True(res1.DataArray[0] == 10);
            Assert.True(res1.Shape.SequenceEqual(new int []{1}));

            Assert.True(res2.Shape.SequenceEqual(new int[] { 1 }));
            Assert.True(res2.DataArray.SequenceEqual(new double[] {10}));

            Assert.True(res3.Shape.SequenceEqual(new int[] { 4 }));
            Assert.True(res3.DataArray.SequenceEqual(new double[] { 1,2,3,4}));

            Assert.True(res4.DataArray[0] == 10);
            Assert.True(res4.Shape.SequenceEqual(new int []{1,1}));

            Assert.True(res5.Shape.SequenceEqual(new int[] { 1,1 }));
            Assert.True(res5.DataArray.SequenceEqual(new double[] {10}));

            Assert.True(res6.Shape.SequenceEqual(new int[] { 4,1 }));
            Assert.True(res6.DataArray.SequenceEqual(new double[] { 1,2,3,4}));

            Assert.Throws<InvalidOperationException>(() => a1.Sum(axis: 2));

     

        }

        [Fact]
        public void NDimArray_Test_Sum_Limit_Axes2()
        {

            NDimArray a1 = new NDimArray(new int[]{1,4},1,2,3,4);

            NDimArray res1 = a1.Sum();
            NDimArray res2 = a1.Sum(0);
            NDimArray res3 = a1.Sum(1);

            NDimArray res4 = a1.Sum(keepDims:true);
            NDimArray res5 = a1.Sum(0,true);
            NDimArray res6 = a1.Sum(1,true);
            

            Assert.True(res1.DataArray[0] == 10);
            Assert.True(res1.Shape.SequenceEqual(new int []{1}));

            Assert.True(res2.Shape.SequenceEqual(new int[] { 4 }));
            Assert.True(res2.DataArray.SequenceEqual(new double[] { 1,2,3,4}));

            Assert.True(res3.Shape.SequenceEqual(new int[] { 1 }));
            Assert.True(res3.DataArray.SequenceEqual(new double[] {10}));

            Assert.True(res4.DataArray[0] == 10);
            Assert.True(res4.Shape.SequenceEqual(new int []{1,1}));

            Assert.True(res5.Shape.SequenceEqual(new int[] { 1,4 }));
            Assert.True(res5.DataArray.SequenceEqual(new double[] { 1,2,3,4}));

            Assert.True(res6.Shape.SequenceEqual(new int[] { 1,1 }));
            Assert.True(res6.DataArray.SequenceEqual(new double[] { 10}));

            Assert.Throws<InvalidOperationException>(() => a1.Sum(axis: 2));


        }

        [Fact]
        public void NDimArray_Test_Tanh()
        {
            NDimArray a1 = new NDimArray(new int[] { 3 }, 1, 2, 3);
            NDimArray a2 = NDimArray.Tanh(a1);


            Assert.True(a2.DataArray.SequenceEqual(new double[] { 0.7615941559557649, 0.9640275800758169, 0.9950547536867305 }));
        }


        [Fact]
        public void NDimArray_Test_Exp()
        {
            NDimArray a1 = new NDimArray(new int[] { 3 }, 1, 2, 3);
            NDimArray a2 = NDimArray.Exp(a1);

            Assert.True(a2.DataArray.SequenceEqual(new double[] { 2.718281828459045, 7.38905609893065, 20.085536923187668 }));
        }


        [Fact]
        public void NDimArray_Test_Transpose()
        {
            NDimArray a1 = new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6);
            NDimArray aT = a1.Transpose();

            Assert.True(aT.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(aT.DataArray.SequenceEqual(new double[] { 1, 3, 5, 2, 4, 6 }));


        }

        [Fact]
        public void NDimArray_Test_Transpose_Dim1()
        {
            NDimArray a1 = new NDimArray(new int[] { 3 }, 1, 2, 3);
            NDimArray aT = a1.Transpose();

            Assert.True(aT.Shape.SequenceEqual(new int[] { 3 }));
            Assert.True(aT.DataArray.SequenceEqual(new double[] { 1, 2, 3 }));


        }

        [Fact]
        public void NDimArray_Test_Transpose_DimException()
        {
            NDimArray a1 = new NDimArray(new int[] { 1, 1, 3 }, 1, 2, 3);

            Assert.Throws<NotImplementedException>(() => a1.Transpose());


        }

        [Fact]
        public void NDimArray_Test_Extend2DArrayByShape()
        {
            NDimArray a1 = new NDimArray(new int[] { 1, 3 }, 1, 2, 3);
            NDimArray a2 = new NDimArray(new int[] { 3 }, 1, 2, 3);
            NDimArray a3 = new NDimArray(new int[] { 4, 1 }, 1, 2, 3, 4);
            int[] newShape = new int[] { 4, 3 };

            NDimArray a1Extended = NDimArray.Extend2DArrayByShape(a1, newShape);
            NDimArray a2Extended = NDimArray.Extend2DArrayByShape(a2, newShape);
            NDimArray a3Extended = NDimArray.Extend2DArrayByShape(a3, newShape);

            Assert.True(a1Extended.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a1Extended.DataArray.SequenceEqual(new double[] { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 }));

            Assert.True(a2Extended.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a2Extended.DataArray.SequenceEqual(new double[] { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 }));

            Assert.True(a3Extended.Shape.SequenceEqual(new int[] { 4, 3 }));
            Assert.True(a3Extended.DataArray.SequenceEqual(new double[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 }));

        }


    }
}

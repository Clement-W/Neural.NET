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
            arr2 = new NDimArray(new int[]{2,3,2}, 1,2,3,4,5,6,7,8,9,10,11,12);
            arr3 = new NDimArray(1);

        }

        [Fact]
        public void NDimArray_Test_Shape()
        {
            //Console.WriteLine("[{0}]", string.Join(", ", arr.Shape));
            //Console.WriteLine(arr.Shape.GetType());
            Assert.True(arr1.Shape.SequenceEqual(new int[] { 3, 4 }));
            Assert.True(arr2.Shape.SequenceEqual(new int[] { 2,3,2 }));
            Assert.True(arr3.Shape.SequenceEqual(new int[] {1}));
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
            Assert.True(arr1.StepIndexes.SequenceEqual(new int[]{4,1}));
            Assert.True(arr2.StepIndexes.SequenceEqual(new int[]{6,2,1}));
            Assert.True(arr3.StepIndexes.SequenceEqual(new int[]{1}));
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
            Assert.True(arr2[1, 2,0] == 10);
            arr3[0] = 10;
            Assert.True(arr3[0] == 10);
        }

        [Fact]
        public void NDimArray_Test_PrintAsMatrix()
        {
            arr1.FillWithValue(5);
            arr1[1,1] = 0;
            arr1.PrintAsMatrix();

            arr2.FillWithValue(1.129);   
            Assert.Throws<InvalidOperationException>(() =>arr2.PrintAsMatrix());
            Assert.Throws<InvalidOperationException>(() =>arr3.PrintAsMatrix());
        }

      



    }
}

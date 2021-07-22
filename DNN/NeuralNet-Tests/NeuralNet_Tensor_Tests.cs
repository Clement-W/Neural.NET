using Xunit;
using NeuralNet.Autodiff;
using System;
using System.Linq;

/*TODO: writer better tests, more files with better names : methodName_scenario_expectedBehavior
ex: Add_SingleNumber_ReturnsSameNumber

Organisation, Action, Assertion est un modèle courant pour les tests unitaires. Comme son nom l’indique, il comporte trois actions principales :

    Organisation, création et configuration des objets selon les besoins
    Agir sur un objet.
    Assertion de ce qui est prévu

Durant l’écriture des tests, essayez d’inclure uniquement une instruction Assert par test. Voici les approches courantes pour utiliser une seule assertion :

    Créez un test distinct pour chaque assertion.
    Utilisez des tests paramétrables.
CAR : 
    En cas d’échec d’une instruction Assert, les assertions qui suivent ne sont pas évaluées.
    Permet de vérifier que vous n’effectuez pas l’assertion de plusieurs cas dans vos tests.
    Vous donne une idée complète des causes de l’échec des tests.
    ->Il existe une exception usuelle à cette règle : l’assertion d’un objet par différenciation. Dans ce cas, il est généralement acceptable d’avoir plusieurs assertions sur chaque propriété pour vérifier que l’objet se trouve dans l’état prévu.
-> faire une méthode et utiliser inline :
[InlineData(null)]
[InlineData("a")]
public void Add_InputNullOrAlphabetic_ThrowsArgumentException(string input)

*/
namespace NeuralNet.UnitTests
{
    public class NeuralNet_Tensor_Tests
    {

        [Fact]
        public void Tensor_Test_Construcors()
        {

            NDimArray a = new NDimArray(new int[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            NDimArray b = new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6);
            Tensor t1 = new Tensor(a);
            Tensor t2 = new Tensor(a, requiresGrad: true);
            TensorDependency[] tdeps = new TensorDependency[] { new TensorDependency(t2, ((w) => w)) };
            Tensor t4 = new Tensor(b, requiresGrad: true, dependencies: tdeps);

            Tensor t5 = new Tensor(new int[] { 3, 2 }, requiresGrad: true);
            Tensor t6 = new Tensor(requiresGrad: true, 1, 2, 3, 4, 5);

            Assert.True(t4.Grad.Shape.SequenceEqual(new int[] { 3, 2 }));
            Assert.True(t2.NDim == 3);
            Assert.True(t5.NDim == 2);
            Assert.True(t6.Data.NbElements == 5);
            Assert.True(t4.Grad.Data.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0 }));
            Assert.True(t1.Grad == null);

            //Console.WriteLine(t4);
        }

        [Fact]
        public void Tensor_Test_Simple_Backward()
        {
            NDimArray b = new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6);

            Tensor t4 = new Tensor(b, requiresGrad: true);
            Assert.True(t4.Grad.Data.DataArray.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0 }));
            t4.Backward(new Tensor(NDimArray.Ones_like(b)));
            Assert.True(t4.Grad.Data.DataArray.SequenceEqual(new double[] { 1, 1, 1, 1, 1, 1 }));

        }

        [Fact]
        public void Tensor_Test_GetBroadcastedShape()
        {
            int[] b1 = NDimArray.GetBroadcastedShapes(new int[] { 2, 3, 1, 4, 2 }, new int[] { 3, 1, 2 });
            Assert.True(b1.SequenceEqual(new int[] { 2, 3, 3, 4, 2 }));

            int[] b2 = NDimArray.GetBroadcastedShapes(new int[] { 2, 3, 1, 4, 2 }, new int[] { 1, 1 });
            Assert.True(b2.SequenceEqual(new int[] { 2, 3, 1, 4, 2 }));

            int[] b3 = NDimArray.GetBroadcastedShapes(new int[] { 2, 3 }, new int[] { 3 });
            Assert.True(b3.SequenceEqual(new int[] { 2, 3 }));

            int[] b4 = NDimArray.GetBroadcastedShapes(new int[] { 2, 3 }, new int[] { 2, 1 });
            Assert.True(b4.SequenceEqual(new int[] { 2, 3 }));

        }

        [Fact]
        public void Tensor_Test_Sum()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = t1.Sum();

            t2.Backward();
            Assert.True(t1.Data.DataArray.SequenceEqual(new double[] { 1, 2, 3 }));
            Assert.True(t2.Data.DataArray.SequenceEqual(new double[] { 6 }));

            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 1, 1, 1 }));

            t2.ZeroGrad();
            t1.ZeroGrad();

            t2.Backward(new Tensor(-5));

            Assert.True(t2.Data.DataArray.SequenceEqual(new double[] { 6 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { -5, -5, -5 }));

        }

        [Fact]
        public void Tensor_Test_Simple_Addition()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(4, 5, 6);

            Tensor t3 = t1 + t2;

            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 5, 7, 9 }));

            t3.Backward(new Tensor(10, 20, 30));

            Assert.True(t3.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t2.Grad == null);

        }

        [Fact]
        public void Tensor_Test_Simple_IAdd()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(4, 5, 6);

            Tensor t3 = t1 + t2;

            t3.Backward(new Tensor(10, 20, 30));

            t1 += new Tensor(0.5);
            Assert.True(t1.Data.DataArray.SequenceEqual(new double[] { 1.5, 2.5, 3.5 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 0, 0, 0 }));

        }

        [Fact]
        public void Tensor_Test_Addition_Broadcast_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 2, 3, 4, 5, 6), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(requiresGrad: true, 7, 8, 9); //shape 3

            Tensor t3 = t1 + t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 8, 10, 12, 11, 13, 15 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 1, 1, 2, 2, 2)));

            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 1, 1, 1, 2, 2, 2 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { 3, 3, 3 }));
        }

        [Fact]
        public void Tensor_Test_Addition_Broadcast_No_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 2, 3, 4, 5, 6), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(new NDimArray(new int[] { 1, 3 }, 7, 8, 9), requiresGrad: true); //shape 1,3

            Tensor t3 = t1 + t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 8, 10, 12, 11, 13, 15 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 1, 1, 2, 2, 2)));

            Assert.True(t1.Grad.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t2.Grad.Shape.SequenceEqual(new int[] { 1, 3 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 1, 1, 1, 2, 2, 2 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { 3, 3, 3 }));
        }


        [Fact]
        public void Tensor_Test_Simple_Multiplication()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(4, 5, 6);

            Tensor t3 = t1 * t2;

            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 4, 10, 18 }));

            t3.Backward(new Tensor(10, 20, 30));

            Assert.True(t3.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 40, 100, 180 }));
            Assert.True(t2.Grad == null);

        }

        [Fact]
        public void Tensor_Test_Simple_IMul()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(4, 5, 6);

            Tensor t3 = t1 * t2;

            t3.Backward(new Tensor(10, 20, 30));

            t1 *= new Tensor(0.5);
            Assert.True(t1.Data.DataArray.SequenceEqual(new double[] { 0.5, 1, 1.5 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 0, 0, 0 }));

        }

        [Fact]
        public void Tensor_Test_Multiplication_Broadcast_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 2, 3, 4, 5, 6), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(requiresGrad: true, 7, 8, 9); //shape 3

            Tensor t3 = t1 * t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 7, 16, 27, 28, 40, 54 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 1, 1, 2, 2, 2)));

            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 7, 8, 9, 14, 16, 18 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { 9, 12, 15 }));
        }

        [Fact]
        public void Tensor_Test_Multiplication_Broadcast_No_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 2, 3, 4, 5, 6), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(new NDimArray(new int[] { 1, 3 }, 7, 8, 9), requiresGrad: true); //shape 1,3

            Tensor t3 = t1 * t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 7, 16, 27, 28, 40, 54 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 1, 1, 2, 2, 2)));

            Assert.True(t1.Grad.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t2.Grad.Shape.SequenceEqual(new int[] { 1, 3 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 7, 8, 9, 14, 16, 18 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { 9, 12, 15 }));
        }


        [Fact]
        public void Tensor_Test_Simple_Negation()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);

            Tensor t3 = -t1;

            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { -1, -2, -3 }));

            t3.Backward(new Tensor(10, 20, 30));

            Assert.True(t3.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { -10, -20, -30 }));

        }


        [Fact]
        public void Tensor_Test_Simple_Substract()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(requiresGrad: true, 4, 5, 6);

            Tensor t3 = t1 - t2;

            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { -3, -3, -3 }));

            t3.Backward(new Tensor(10, 20, 30));

            Assert.True(t3.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -10, -20, -30 }));

        }

        [Fact]
        public void Tensor_Test_Simple_ISub()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(4, 5, 6);

            Tensor t3 = t1 - t2;

            t3.Backward(new Tensor(10, 20, 30));

            t1 -= new Tensor(0.5);
            Assert.True(t1.Data.DataArray.SequenceEqual(new double[] { 0.5, 1.5, 2.5 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 0, 0, 0 }));

        }

        [Fact]
        public void Tensor_Test_Substract_Broadcast_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 2, 3, 4, 5, 6), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(requiresGrad: true, 7, 8, 9); //shape 3

            Tensor t3 = t1 - t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { -6, -6, -6, -3, -3, -3 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 1, 1, 2, 2, 2)));

            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 1, 1, 1, 2, 2, 2 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -3, -3, -3 }));
        }

        [Fact]
        public void Tensor_Test_Substract_Broadcast_No_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 2, 3, 4, 5, 6), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(new NDimArray(new int[] { 1, 3 }, 7, 8, 9), requiresGrad: true); //shape 1,3

            Tensor t3 = t1 - t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { -6, -6, -6, -3, -3, -3 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 1, 1, 1, 2, 2, 2)));

            Assert.True(t1.Grad.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t2.Grad.Shape.SequenceEqual(new int[] { 1, 3 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 1, 1, 1, 2, 2, 2 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -3, -3, -3 }));

        }


        [Fact]
        public void Tensor_Test_Simple_TrueDivision()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(requiresGrad: true, 2, 2, 2);

            Tensor t3 = t1 / t2;

            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 0.5, 1, 1.5 }));

            t3.Backward(new Tensor(10, 20, 30));

            Assert.True(t3.Grad.Data.DataArray.SequenceEqual(new double[] { 10, 20, 30 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 5, 10, 15 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -2.5, -10, -22.5 })); //[10,20,30] * (-t1/t2*t2)
            //[-10,-40,-90]/[4,4,4] = [-10/4,-10,-90/4]

        }

        [Fact]
        public void Tensor_Test_Simple_ITruediv()
        {
            Tensor t1 = new Tensor(requiresGrad: true, 1, 2, 3);
            Tensor t2 = new Tensor(requiresGrad: true, 2, 2, 2);

            Tensor t3 = t1 / t2;

            t3.Backward(new Tensor(10, 20, 30));


            t1 /= new Tensor(0.5);
            Assert.True(t1.Data.DataArray.SequenceEqual(new double[] { 2, 4, 6 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 0, 0, 0 }));

        }

        [Fact]
        public void Tensor_Test_TrueDivision_Broadcast_New_Dimension()
        {
            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 2, 4, 6, 8, 10, 12), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(requiresGrad: true, 2, 2, 2); //shape 3

            Tensor t3 = t1 / t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 1, 2, 3, 4, 5, 6 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 4, 4, 4, 10, 10, 10)));

            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 2, 2, 2, 5, 5, 5 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -22, -29, -36 }));
            // [[-0.5,-1,-1.5],[-2,-2.5,-3]] * grad
            // [[-2,-4,-6],[-20,-25,-30]] = [-22,-29,-36]
        }

        [Fact]
        public void Tensor_Test_TrueDivision_Broadcast_No_New_Dimension()
        {

            Tensor t1 = new Tensor(new NDimArray(new int[] { 2, 3 }, 2, 4, 6, 8, 10, 12), requiresGrad: true); //shape 2,3
            Tensor t2 = new Tensor(new NDimArray(new int[] { 1, 3 }, 2, 2, 2), requiresGrad: true); //shape 1,3


            Tensor t3 = t1 / t2;
            Assert.True(t3.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 1, 2, 3, 4, 5, 6 }));

            t3.Backward(new Tensor(new NDimArray(new int[] { 2, 3 }, 4, 4, 4, 10, 10, 10)));

            Assert.True(t1.Grad.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t2.Grad.Shape.SequenceEqual(new int[] { 1, 3 }));
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { 2, 2, 2, 5, 5, 5 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -22, -29, -36 }));
            // [[-0.5,-1,-1.5],[-2,-2.5,-3]] * grad
            // [[-2,-4,-6],[-20,-25,-30]] = [-22,-29,-36]

        }


        [Fact]
        public void Tensor_Test_Matmul()
        {

            Tensor t1 = new Tensor(new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6), requiresGrad: true);
            Tensor t2 = new Tensor(new NDimArray(new int[] { 2, 1 }, 10, 20), requiresGrad: true);


            Tensor t3 = Tensor.Matmul(t1, t2);

            Assert.True(t3.Shape.SequenceEqual(new int[] { 3, 1 }));
            Assert.True(t3.Data.DataArray.SequenceEqual(new double[] { 50, 110, 170 }));
        }

        [Fact]
        public void Tensor_Test_Matmul_After_Backward()
        {

            Tensor t1 = new Tensor(new NDimArray(new int[] { 3, 2 }, 1, 2, 3, 4, 5, 6), requiresGrad: true);
            Tensor t2 = new Tensor(new NDimArray(new int[] { 2, 1 }, 10, 20), requiresGrad: true);

            Tensor t3 = Tensor.Matmul(t1, t2);

            NDimArray grad = new NDimArray(new int[] { 3, 1 }, -1, -2, -3);
            t3.Backward(new Tensor(grad));

            Assert.True(t1.Grad.Shape.SequenceEqual(new int[] { 3, 2 }));
            Assert.True(t2.Grad.Shape.SequenceEqual(new int[] { 2, 1 })); //could be 2,1
            Assert.True(t1.Grad.Data.DataArray.SequenceEqual(new double[] { -10, -20, -20, -40, -30, -60 }));
            Assert.True(t2.Grad.Data.DataArray.SequenceEqual(new double[] { -22, -28 }));

        }

        [Fact]
        public void Tensor_Test_Minimize_Function()
        {
            Tensor x = new Tensor(requiresGrad: true, 29, 63, 3, 9.6, -77, -23);

            // Try to minimize the sum of squares

            Tensor lr = new Tensor(0.1);

            for (int i = 0; i < 100; i++)
            {
                x.ZeroGrad();
                Tensor sumOfSquares = (x * x).Sum();

                sumOfSquares.Backward();

                Tensor deltaX = lr * x.Grad;

                x -= deltaX;
            }
            Assert.True(Math.Round(x.Sum().Data.DataArray[0]) == 0);
        }

        [Fact]
        public void Tensor_Test_Simple_Slice()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            Tensor t1 = new Tensor(a1);
            Tensor t2 = t1.Slice2DTensor(2, 4);

            Assert.True(t2.Shape.SequenceEqual(new int[] { 2, 3 }));
            Assert.True(t2.Data.DataArray.SequenceEqual(new double[] { 7, 8, 9, 10, 11, 12 }));

        }

        
        [Fact]
        public void Tensor_Test_Slice_Backward()
        {
            NDimArray a1 = new NDimArray(new int[] { 4, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            Tensor t1 = new Tensor(a1,requiresGrad:true);
            Tensor t2 = t1.Slice2DTensor(2, 4); // contain 7, 8, 9, 10, 11, 12

            Tensor grad = new Tensor(requiresGrad:false,new int[] { 2, 3 }, 1,1,1,1,1,1);  
     
            t2.Backward(grad);     

                  
        }




    }
}

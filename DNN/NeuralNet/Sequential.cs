using NeuralNet.Autodiff;
namespace NeuralNet
{

    /// <summary>
    /// This class represents a sequential model, a stack of layers, and/or activation functions
    /// </summary>
    public class Sequential : Model
    {
        /// <summary>
        /// The blocks that compose the model
        /// </summary>
        /// <value></value>
        public IBlock[] Blocks{get;private set;}

        /// <summary>
        /// Constructor used to create the sequential model
        /// </summary>
        /// <param name="blocks"></param>
        public Sequential(params IBlock[] blocks){
            Blocks = blocks;
        }

        /// <summary>
        /// Predict the output of the neural network by feeding it with the input
        /// </summary>
        /// <param name="inputs">The input tensor</param>
        /// <returns>Output of the neural network</returns>
        public override Tensor Predict(Tensor inputs){
            Tensor output = inputs;
            foreach(IBlock block in Blocks){
                output = block.Forward(output);
            }
            return output;
        }

        public override string ToString()
        {
            string res="";
            int cpt=0;
            foreach(IBlock block in Blocks){
                res+=cpt++ + " : " + block.GetType().Name;
                if(block.GetType() == typeof(LinearLayer)){
                    res+= $" (input size={(block as LinearLayer).InputSize}, output size={(block as LinearLayer).OutputSize})";
                }
                res+="\n";
            }
            return res;
        }


    }
}
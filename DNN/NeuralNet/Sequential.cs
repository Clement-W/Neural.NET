using NeuralNet.Autodiff;
namespace NeuralNet
{
    public class Sequential : Model
    {
        public IBlock[] Blocks{get;private set;}
        public Sequential(params IBlock[] blocks){
            Blocks = blocks;
        }

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
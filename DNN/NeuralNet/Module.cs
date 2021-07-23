using System.Reflection;
using System;
using System.Collections.Generic;
namespace NeuralNet
{
    // A module represents a collection of Parameters
    public class Module
    {

        // Return every properties of type Parameter or Module
        public IEnumerable<Parameter> Parameters()
        { //TODO: This method is not efficient at all. Would it be better with an array instead of using Reflection ?

            // If the module is a sequential class, get the parameters in it's list of blocks
            if (this.GetType() == typeof(Sequential))
            {
                foreach (IBlock block in (this as Sequential).Blocks)
                {
                    // If the block is a module, get it's parameters
                    if (block.GetType().IsSubclassOf(typeof(Module)) || block.GetType() == typeof(Module))
                    {
                        IEnumerable<Parameter> submoduleParameters = (block as Module).Parameters();
                        if (submoduleParameters != null)
                        {
                            foreach (Parameter param in submoduleParameters)
                            {
                                yield return param;
                            }
                        }
                    }
                }
            }
            //else, get the parameters in it's properties
            else
            {

                PropertyInfo[] properties = this.GetType().GetProperties();
                // Foreach of the properties, get the Parameter and Module properties to return every Parameter properties
                foreach (PropertyInfo property in properties)
                {
                    if (property.PropertyType == typeof(Parameter))
                    {
                        yield return property.GetValue(this) as Parameter;
                    }
                    else if (property.PropertyType.IsSubclassOf(typeof(Module)) || property.PropertyType == typeof(Module))
                    {
                        // Return the parameters of the module by calling Parameters() on it
                        IEnumerable<Parameter> moduleParamereters = (property.GetValue(this) as Module).Parameters();
                        if (moduleParamereters != null)
                        {
                            foreach (Parameter param in moduleParamereters)
                            {
                                yield return param;
                            }
                        }
                    }
                }
            }
        }


        // Set the gradient of every parameters to 0
        public void ZeroGrad()
        {
            foreach (Parameter param in this.Parameters())
            {
                param.ZeroGrad();
            }
        }

        public override string ToString()
        {
            string ret = "Parameters : \n";
            foreach (Parameter param in this.Parameters())
            {
                ret += param.ToString() + "\n";
            }
            return ret;
        }
    }



}


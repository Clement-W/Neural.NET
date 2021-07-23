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

            PropertyInfo[] properties = this.GetType().GetProperties();
            // Foreach of the properties, get the Parameter and Module ones to return every Parameter properties
            foreach (PropertyInfo property in properties)
            {
                if (property.PropertyType == typeof(Parameter))
                {
                    yield return property.GetValue(this) as Parameter;
                }
                else if (property.PropertyType.IsSubclassOf(typeof(Module)) || property.PropertyType == typeof(Module))
                {
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
            string ret="Parameters : \n";
            foreach(Parameter param in this.Parameters()){
                ret+= param.ToString() + "\n";
            }
            return ret;
        }
    }



}


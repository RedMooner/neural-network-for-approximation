using System.Collections.Generic;

namespace NN
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType Type;
        public IActivationFunction ActivationFunction;

        public Layer(List<Neuron> neurons, IActivationFunction activationFunction, NeuronType type = NeuronType.Normal)
        {
            // TODO: проверить все входные нейроны на соответствие типу
            Neurons = neurons;
            Type = type;
            ActivationFunction = activationFunction;
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}

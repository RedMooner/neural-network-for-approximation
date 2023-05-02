using System;
using System.Collections.Generic;

namespace NN
{
    public class Neuron
    {
        public IActivationFunction ActivationFunction { get; }
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }

        public Neuron(int inputCount, IActivationFunction activationFunction, NeuronType neuronType = NeuronType.Normal)
        {
            Inputs = new List<double>();
            Weights = new List<double>();
            NeuronType = neuronType;
            ActivationFunction = activationFunction;

            InitWeightsRandomValue(inputCount);
        }
        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * ActivationFunction.Derrative(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeigth = weight - input * Delta * learningRate;
                Weights[i] = newWeigth;
            }
        }
        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }
            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = ActivationFunction.Activation(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }
        public void SetWeights(List<double> doubles)
        {
            Weights.Clear();
            for (int i = 0; i < doubles.Count; i++)
            {
                Weights.Add(doubles[i]);
            }
        }
    }
}

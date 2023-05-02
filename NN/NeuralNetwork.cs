﻿using System.Collections.Generic;
using System.Linq;

namespace NN
{
    public class NeuralNetwork
    {
        public List<Layer> Layers;
        public Topology Topology { get; }
        public NeuralNetwork(Topology topology, params IActivationFunction[] activationFunctions)
        {
            Topology = topology;

            Layers = new List<Layer>();
            if(activationFunctions.Length != 3)
                throw new System.ArgumentException("Error, activations function must be a three");

            CreateInputLayer(activationFunctions[0]);
            CreateHiddenLayers(activationFunctions[1]);
            CreateOutputLayer(activationFunctions[2]);
        }
        public Neuron Predict(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSingals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSingals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }
        private void CreateOutputLayer(IActivationFunction function)
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, function, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, function, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers(IActivationFunction function)
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount, function);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons, function, NeuronType.Normal);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateInputLayer(IActivationFunction function)
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, function, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var hiddenLayer = new Layer(inputNeurons, function);
            Layers.Add(hiddenLayer);
        }
    }
}

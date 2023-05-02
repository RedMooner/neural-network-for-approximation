using System;

namespace NN
{
    internal class Program
    {
        //Console.WriteLine(string.Join(",", neuralNetwork.Layers[1].Neurons[0].Output));
        //Console.WriteLine(string.Join(",", neuralNetwork.Layers[1].Neurons[1].Output));

        //Console.WriteLine(string.Join(",", neuralNetwork.Layers[0].Neurons[0].Output));
        //Console.WriteLine(string.Join(",", neuralNetwork.Layers[0].Neurons[1].Output));
        static void Main(string[] args)
        {
            Sigmoid sigmoid = new Sigmoid();

            var topology = new Topology(3, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology, sigmoid, sigmoid, sigmoid);

            neuralNetwork.Layers[1].Neurons[0].SetWeights(new System.Collections.Generic.List<double>() { 0.25, 0.25, 0 });
            neuralNetwork.Layers[1].Neurons[1].SetWeights(new System.Collections.Generic.List<double>() { 0.5, -0.4, 0.9 });
            neuralNetwork.Layers[2].Neurons[0].SetWeights(new System.Collections.Generic.List<double>() { -1, 1 });
            neuralNetwork.Layers[0].Neurons[0].SetWeights(new System.Collections.Generic.List<double>() { 1, 1, 1 });
            neuralNetwork.Layers[0].Neurons[1].SetWeights(new System.Collections.Generic.List<double>() { 1, 1, 1 });

            Console.Write(neuralNetwork.Predict(1, 1, 0).Output);

            Console.ReadLine();
        }

    }
    public class Sigmoid : IActivationFunction
    {
        public double Activation(double x)
        {
            if (x > 0.5)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        public double Derrative(double x)
        {
            throw new System.NotImplementedException();
        }
    }
    public class Linear : IActivationFunction
    {
        public double Activation(double x)
        {
            return x;
        }

        public double Derrative(double x)
        {
            throw new System.NotImplementedException();
        }
    }
}

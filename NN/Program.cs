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
            Linear linearlinear = new Linear(); 
            var topology = new Topology(3, 1, 0.01, 2);
            var neuralNetwork = new NeuralNetwork(topology, sigmoid, sigmoid, sigmoid);

            //neuralNetwork.Layers[1].Neurons[0].SetWeights(new System.Collections.Generic.List<double>() { 0.25, 0.25, 0 });
            //neuralNetwork.Layers[1].Neurons[1].SetWeights(new System.Collections.Generic.List<double>() { 0.5, -0.4, 0.9 });
            //neuralNetwork.Layers[2].Neurons[0].SetWeights(new System.Collections.Generic.List<double>() { -1, 1 });
            //neuralNetwork.Layers[0].Neurons[0].SetWeights(new System.Collections.Generic.List<double>() { 1, 1, 1 });
            //neuralNetwork.Layers[0].Neurons[1].SetWeights(new System.Collections.Generic.List<double>() { 1, 1, 1 });

            var expected = new double[8] { 0, 1, 0, 0, 1, 1, 0, 1 };
            var inputs = new double[,]
            {
                {0,0,0 },
                {0,0,1 },
                {0,1,0 },
                {0,1,1 },
                {1,0,0 },
                {1,0,1 },
                {1,1,0 },
                {1,1,1 },
            };
            neuralNetwork.Learn(expected, inputs, 5000);

            Console.WriteLine(neuralNetwork.Predict(0, 0, 0).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(0, 0, 1).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(0, 1, 0).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(0, 1, 1).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(1, 0, 0).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(1, 0, 1).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(1, 1, 0).Output > 0.5);
            Console.WriteLine(neuralNetwork.Predict(1, 1, 1).Output > 0.5);

            Console.ReadLine();
        }

    }
    public class Sigmoid : IActivationFunction
    {
        public double Activation(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public double Derrative(double x)
        {
            return this.Activation(x) * (1-this.Activation(x));
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
            return x;
        }
    }
}

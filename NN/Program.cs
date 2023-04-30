using System;

namespace NN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var topology = new Topology(3, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);
            Console.Write(neuralNetwork.Predict(0, 1, 1).Output);
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
}

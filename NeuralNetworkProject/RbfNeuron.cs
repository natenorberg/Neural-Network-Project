using System;
using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public class RbfNeuron : Neuron
	{
		private double _output;

		public RbfNeuron(double[] center, double spread)
		{
			Center = center;
			Spread = spread;

			Inputs = new List<Connection>();
			Outputs = new List<Connection>();
		}

		/// <summary>
		/// Gets or sets the center of the gaussian.
		/// </summary>
		public double[] Center { get; set;}

		/// <summary>
		/// Gets or sets the spread.
		/// </summary>
		public double Spread { get; set; }

		/// <summary>
		/// Evaluate this neuron.
		/// </summary>
		public override void Evaluate()
		{
			_output = GaussianFunction(Inputs);

			foreach (var output in Outputs)
			{
				output.Value = _output;
			}
		}

		private double GaussianFunction(List<Connection> inputs)
		{
			if (inputs.Count != Center.Length) {
				throw new System.NotSupportedException("Wrong number of inputs");
			}

			var difference = new double[inputs.Count];

			for (int i=0; i<inputs.Count; i++)
			{
				difference[i] = inputs[i].Value - Center[i];
			}

			// Find the Euclidean Norm
			double norm = 0;
			foreach (var value in difference)
			{
				norm += Math.Pow(value, 2);
			}

			return Math.Pow(Math.E, (-1 * norm) / (2 * Math.Pow(Spread, 2)));
		}
	}
}


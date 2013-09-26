using System;

namespace NeuralNetworkProject
{
	public class Connection
	{
		public Connection (double input, double weight)
		{
			Value = input;
			Weight = weight;
		}

		public Connection (double weight)
		{
			Weight = weight;
		}

		public Connection()
		{
			var random = new Random();
			Weight = random.NextDouble();
		}

		/// <summary>
		/// Value being passed through
		/// </summary>
		public double Value { get; set; }

		/// <summary>
		/// The weight of the connection
		/// </summary>
		public double Weight { get; set;}

		/// <summary>
		/// Gets or sets the momentum of the weight.
		/// </summary>
		public double Momentum {get; set; }
	}
}


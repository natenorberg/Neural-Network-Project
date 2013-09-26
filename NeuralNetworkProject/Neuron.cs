using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public abstract class Neuron
	{
		/// <summary>
		/// The neuron's inputs
		/// </summary>
		public List<Connection> Inputs {get; set;}

		/// <summary>
		/// The neuron's outputs
		/// </summary>
		public List<Connection> Outputs { get; set;}

		/// <summary>
		/// Evaluate this instance.
		/// </summary>
		public abstract void Evaluate();
	}
}


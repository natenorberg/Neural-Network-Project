using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public abstract class FeedForwardNetwork : NeuralNetwork
	{
		/// <summary>
		/// Passes number of ins and outs to the base constructor
		/// </summary>
		protected FeedForwardNetwork(int numInputs, int numOutputs) : base(numInputs, numOutputs)
		{
		}

		/// <summary>
		/// Layers of neurons
		/// </summary>
		public List<IEnumerable<Neuron>> Layers { get; set; }

		/// <summary>
		/// Runs the network.
		/// </summary>
		public override IEnumerable<double> RunNetwork(IEnumerable<double> inputVector)
		{
			// Feed values into inputs
			var inputData = inputVector as List<double>;
			int i=0;

			foreach (var input in Inputs)
			{
				input.Value = inputData[i];
				i++;
			}

			// Feed values forward through the network
			foreach (var layer in Layers)
			{
				foreach (var neuron in layer)
				{
					neuron.Evaluate();
				}
			}

			// Build output list
			var outputList = new List<double>();

			foreach (var output in Outputs)
			{
				outputList.Add(output.Value);
			}

			return outputList;
		}
	}
}


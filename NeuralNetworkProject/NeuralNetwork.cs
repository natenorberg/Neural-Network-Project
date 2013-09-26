using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public abstract class NeuralNetwork
	{
		protected NeuralNetwork(int numInputs, int numOutputs)
		{
			CreateInputArray(numInputs);
			CreateOutputArray(numOutputs);
			Neurons = new List<Neuron>();
		}

		/// <summary>
		/// Inputs into the network
		/// </summary>
		public IEnumerable<Connection> Inputs {get; set;}

		/// <summary>
		/// Outputs from the network
		/// </summary>
		public IEnumerable<Connection> Outputs { get; set;}

		/// <summary>
		/// The neurons in the network
		/// </summary>
		public List<Neuron> Neurons { get; set; }

		void CreateInputArray (int numInputs)
		{
			Inputs = new List<Connection>();
			var inputList = Inputs as List<Connection>;

			for (int i=0; i<numInputs; i++)
			{
				Connection newInput = new Connection(1);
				inputList.Add(newInput);
			}
		}

		void CreateOutputArray (int numOutputs)
		{
			Outputs = new List<Connection>();
			var outputList = Outputs as List<Connection>;

			for (int i=0; i<numOutputs; i++)
			{
				Connection newOutput = new Connection(1);
				outputList.Add(newOutput);
			}
		}

		/// <summary>
		/// Runs the network.
		/// </summary>
		public abstract IEnumerable<double> RunNetwork(IEnumerable<double> input);

		/// <summary>
		/// Updates the network.
		/// </summary>
		public abstract void UpdateNetwork(IEnumerable<double> expectedOutput);

		/// <summary>
		/// Prints the weights.
		/// </summary>
		public abstract void PrintWeights();
	}
}


using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public class BiasNode : Neuron
	{
		public BiasNode()
		{
			Outputs = new List<Connection>();
		}

		public override void Evaluate()
		{
			foreach (var output in Outputs)
			{
				output.Value = 1.0;
			}
		}
	}
}


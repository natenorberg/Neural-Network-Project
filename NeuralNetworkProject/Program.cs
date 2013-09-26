namespace NeuralNetworkProject
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			ITunableParameterService parameters = TunableParameterService.Instance;
			int numInputs = parameters.NumberOfInputs;
			int numOutputs = parameters.NumberOfOutputs;
			NetworkType type = parameters.NeuralNetworkType;
			int numHiddenLayers = parameters.NumberOfHiddenLayers;
			int neuronsPerHiddenLayer = parameters.NumberOfNodesPerHiddenLayer;

			FunctionApproximator approximator = new FunctionApproximator();
			NeuralNetwork network;

			if (type == NetworkType.MLP) {
				network = new MultiLayerPerceptron(numInputs, numOutputs, numHiddenLayers, neuronsPerHiddenLayer);
			}
			else {
				network = new RadialBasisFunctionNetwork(numInputs, numOutputs);
			}

			approximator.ApproximateFunction(network);
		}
	}

	public enum NetworkType
	{
		MLP,
		RBF
	}

}

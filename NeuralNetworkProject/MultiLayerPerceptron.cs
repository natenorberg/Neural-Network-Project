using System.Collections.Generic;
using System;
using System.IO;
using NeuralNetworkProject;

namespace NeuralNetworkProject
{
	public class MultiLayerPerceptron : FeedForwardNetwork
	{
		int _numHiddenLayers;
		int _numNeuronsPerHiddenLayer;
		bool _isMomentumUsed;
		double _momentumAmount;
		double _learningRate;
		double _sigmoidAlpha;
		bool _isAnnealingUsed;
		double _annealingValue;

		public MultiLayerPerceptron(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer)
			:this(numInputs, numOutputs, numHiddenLayers, numNeuronsPerHiddenLayer, TunableParameterService.Instance)
		{
		}

		public MultiLayerPerceptron(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, ITunableParameterService paramService)
			:base(numInputs, numOutputs)
		{
			_numHiddenLayers = numHiddenLayers;
			_numNeuronsPerHiddenLayer = numNeuronsPerHiddenLayer;
			_isMomentumUsed = paramService.IsMomentumUsed;
			_momentumAmount = paramService.MomentumAmount;
			_learningRate = paramService.LearningRate;
			_sigmoidAlpha = paramService.SigmoidAlpha;
			_isAnnealingUsed = paramService.IsAnnealingUsed;
			_annealingValue = paramService.AnnealingValue;

			BuildNetwork();
		}

		public override void UpdateNetwork(IEnumerable<double> outputVector)
		{
			var expectedOutputs = outputVector as List<double>;
			if (expectedOutputs == null)
			{
				throw new InvalidDataException("Expected values should be a list");
			}

			// Update the gradient values first

			// Output layer
			var outputLayer = Layers[_numHiddenLayers + 1] as List<Neuron>;
			for (int i=0; i<expectedOutputs.Count; i++)
			{
				var outputNode = outputLayer[i] as MlpNeuron;

				// Outputs have a linear activation function so f'(x) = 1
				outputNode.Gradient = (expectedOutputs[i] - outputNode.OutputValue);
			}

			//Hidden layers
			for (int i=_numHiddenLayers; i>=0; i--)
			{
				var hiddenLayer = Layers[i] as List<Neuron>;
				var nextLayer = Layers[i + 1] as List<Neuron>;

				foreach (var neuron in hiddenLayer)
				{
					var mlpNeuron = neuron as MlpNeuron;

					if (mlpNeuron != null) 
					{
						double downstreamGradientSum = 0;
						for (int j=0; j<neuron.Outputs.Count; j++) {
							var downstreamNeuron = nextLayer[j] as MlpNeuron;
							if (downstreamNeuron == null) // Skip bias nodes
								continue;

							var connection = neuron.Outputs[j];
							downstreamGradientSum += connection.Weight * downstreamNeuron.OutputValue;
						}

						// Derivative of logistic function: f'(x) = f(x)*(1-f(x))
						mlpNeuron.Gradient = _sigmoidAlpha * mlpNeuron.OutputValue * (1 - mlpNeuron.OutputValue) * downstreamGradientSum;
					}
				}
			}

			// Now update weights
			foreach (var neuron in Neurons)
			{
				var mlpNeuron = neuron as MlpNeuron;
				if (mlpNeuron == null)
					continue;

				foreach (var input in Inputs)
				{
					double change = _learningRate * mlpNeuron.Gradient * input.Value;

					if (_isMomentumUsed)
					{
						change = _momentumAmount * input.Momentum + (1 - _momentumAmount) * change;
						input.Momentum = change;
					}

					input.Weight += change;
				}
			}

			if (_isAnnealingUsed)
			{
				_learningRate -= _annealingValue;
			}

		}

		/// <summary>
		/// Builds the network.
		/// </summary>
		private void BuildNetwork()
		{
			Layers = new List<IEnumerable<Neuron>>();
			Layers.Add(CreateInputLayer());
			Layers.AddRange(CreateHiddenLayers());
			Layers.Add(CreateOutputLayer());

			AddConnections();
		}

		/// <summary>
		/// Creates the input layer.
		/// </summary>
		IEnumerable<Neuron> CreateInputLayer()
		{
			var inputNodes = new List<Neuron>();
			foreach (var input in Inputs)
			{
				var node = new MlpNeuron(new List<Connection>{ input }, new List<Connection>(), FunctionType.Linear);
				inputNodes.Add(node);
				Neurons.Add(node);
			}

			return inputNodes;
		}

		/// <summary>
		/// Creates the hidden layers.
		/// </summary>
		IEnumerable<IEnumerable<Neuron>> CreateHiddenLayers()
		{
			var hiddenLayers = new List<IEnumerable<Neuron>>();

			for (int i=0; i<_numHiddenLayers; i++)
			{
				var hiddenNodes = new List<Neuron>();

				for (int j=0; j<_numNeuronsPerHiddenLayer; j++)
				{
					var neuron = new MlpNeuron(new List<Connection>(), new List<Connection>(), FunctionType.Sigmoid);
					hiddenNodes.Add(neuron);
					Neurons.Add(neuron);
				}

				hiddenNodes.Add(new BiasNode());

				hiddenLayers.Add(hiddenNodes);
			}

			return hiddenLayers;
		}

		/// <summary>
		/// Creates the output layer.
		/// </summary>
		IEnumerable<Neuron> CreateOutputLayer()
		{
			var outputNodes = new List<Neuron>();
			foreach (var output in Outputs)
			{
				var node = new MlpNeuron(new List<Connection>(), new List<Connection>{ output }, FunctionType.Linear);
				outputNodes.Add(node);
				Neurons.Add(node);
			}

			return outputNodes;
		}

		/// <summary>
		/// Adds the connections.
		/// </summary>
		void AddConnections()
		{
			for (int i=0; i<Layers.Count-1; i++)
			{
				var layer = Layers[i];
				var nextLayer = Layers[i + 1];

				foreach (var fromNeuron in layer)
				{
					foreach (var toNeuron in nextLayer)
					{
						var newConnection = new Connection();
						fromNeuron.Outputs.Add(newConnection);

						if (! (toNeuron is BiasNode)) {
							toNeuron.Inputs.Add(newConnection);
						}
					}
				}
			}
		}

		public override void PrintWeights()
		{
			Console.Write("Inputs:\n");
			int i = 0;
			foreach (var input in Inputs)
			{
				Console.Write("Input {0}: {1}\n", i, input.Weight);
			}

			i = 0;
			foreach (var layer in Layers)
			{
				Console.Write("Layer {0}\n", i);
				int j = 0;
				foreach (var neuron in layer)
				{
					int k = 0;
					Console.Write("Neuron[{0}][{1}]:\n", i, j);
					foreach (var output in neuron.Outputs)
					{
						Console.Write("Weight from [{0}][{1}] to [{2}][{3}]: {4}\n", i, j, i + 1, k, output.Weight);
						k++;
					}
					j++;
				}
				i++;
			}
		}
	}
}


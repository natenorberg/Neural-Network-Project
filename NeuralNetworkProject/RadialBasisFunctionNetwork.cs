using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.NetworkInformation;

namespace NeuralNetworkProject
{
	public class RadialBasisFunctionNetwork : FeedForwardNetwork
	{
		int _numCenters;
		int _numDimensions;
		double _lowerBound;
		double _upperBound;
		double _spacing;
		double _spread;
		double _learningRate;
		bool _isMomentumUsed;
		double _momentumAmount;
		bool _isAnnealingUsed;
		double _annealingValue;

		/// <summary>
		/// Constructor
		/// </summary>
		public RadialBasisFunctionNetwork(int numInputs, int numOutputs) : this(numInputs, numOutputs, TunableParameterService.Instance)
		{}

		public RadialBasisFunctionNetwork(int numInputs, int numOutputs, ITunableParameterService paramService)
			:base(numInputs, numOutputs)
		{
			_numCenters = paramService.NumberOfCenters;
			_numDimensions = numInputs;
			_lowerBound = paramService.LowerBound;
			_upperBound = paramService.UpperBound;
			_learningRate = paramService.LearningRate;
			_isMomentumUsed = paramService.IsMomentumUsed;
			_momentumAmount = paramService.MomentumAmount;
			_isAnnealingUsed = paramService.IsAnnealingUsed;
			_annealingValue = paramService.AnnealingValue;

			CalculateSpacingAndSpread();
			BuildNetwork();
		}

		public override void UpdateNetwork(IEnumerable<double> expectedOutput)
		{
			var expectedOutputs = (List<double>)expectedOutput;
			// We only need to update weights on the output node
			var outputLayer = (List<Neuron>)Layers[2];
			for (int i=0; i<outputLayer.Count; i++)
			{
				var outputNode = outputLayer[i];
				var actualOutput = outputNode.Outputs[i].Value;
				foreach (var input in outputNode.Inputs)
				{
					var change = _learningRate * (expectedOutputs[i] - actualOutput) * input.Value;

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
				_learningRate -= _annealingValue * _learningRate;
			}
		}

		/// <summary>
		/// Prints the weights and center information.
		/// </summary>
		public override void PrintWeights()
		{
			var hiddenLayer = (List<Neuron>)Layers[1];

			for (int i=0; i<hiddenLayer.Count-1; i++)
			{
				var basisNode = (RbfNeuron)hiddenLayer[i];
				Console.Write("Mean: {0},{1}  Weight:{2}  Output:{3}\n", 
				              basisNode.Center[0], basisNode.Center[1], basisNode.Outputs[0].Weight, basisNode.Outputs[0].Value);
			}
		}

		private void CalculateSpacingAndSpread()
		{
			_spacing = (_upperBound - _lowerBound) / (Math.Pow(_numCenters, 1.0 / _numDimensions) - 1);

			double maxDistance = Math.Sqrt(2 * Math.Pow(_spacing, 2));
			_spread = maxDistance / _numCenters;

		}

		void BuildNetwork()
		{
			Layers = new List<IEnumerable<Neuron>>();
			Layers.Add(CreateInputLayer());
			Layers.Add(CreateHiddenLayer());
			Layers.Add(CreateOutputLayer());

			AddConnections();
		}

		IEnumerable<Neuron> CreateInputLayer()
		{
			var inputLayer = new List<Neuron>();

			foreach (var input in Inputs)
			{
				var newNeuron = new MlpNeuron(new List<Connection> { input }, new List<Connection>(), FunctionType.Linear);
				inputLayer.Add(newNeuron);
				Neurons.Add(newNeuron);
			}

			return inputLayer;
		}

		IEnumerable<Neuron> CreateHiddenLayer()
		{
			var hiddenLayer = new List<Neuron>();

			int gridSize = (int)Math.Pow(_numCenters, 1.0 / _numDimensions);

			// TODO: Find a less horrible way to do this
			for (int i=0; i<gridSize; i++) 

			{
				for (int j=0; j<gridSize; j++) 
				{
					if (_numDimensions == 2) {
						double[] center = { (i*_spacing)+_lowerBound, (j*_spacing)+_lowerBound};
						var newNeuron = new RbfNeuron(center, _spread);

						hiddenLayer.Add(newNeuron);
						Neurons.Add(newNeuron);
					}
					else {
						for (int k=0; k<gridSize; k++)
						{
							if (_numDimensions == 3)
							{
								double[] center = new double[3]{ (i*_spacing)+_lowerBound, (j*_spacing)+_lowerBound, (k*_spacing+_lowerBound)};
								var newNeuron = new RbfNeuron(center, _spread);

								hiddenLayer.Add(newNeuron);
								Neurons.Add(newNeuron);
							}
							else {
								for (int l=0; l<gridSize; l++)
								{
									if (_numDimensions == 4)
									{
										double[] center = { (i*_spacing)+_lowerBound, (j*_spacing)+_lowerBound, 
											(k*_spacing)+_lowerBound, (l*_spacing)+_lowerBound };
										var newNeuron = new RbfNeuron(center, _spread);

										hiddenLayer.Add(newNeuron);
										Neurons.Add(newNeuron);
									}
									else {
										for (int m=0; m<gridSize; m++)
										{
											double[] center = { (i*_spacing)+_lowerBound, (j*_spacing)+_lowerBound, 
												(k*_spacing)+_lowerBound, (l*_spacing)+_lowerBound, (m*_spacing)+_lowerBound };
											var newNeuron = new RbfNeuron(center, _spread);

											hiddenLayer.Add(newNeuron);
											Neurons.Add(newNeuron);
										}
									}
								}
							}
						}
					}
				}
			}

			hiddenLayer.Add(new BiasNode());

			return hiddenLayer;
		}

		IEnumerable<Neuron> CreateOutputLayer()
		{
			var outputLayer = new List<Neuron>();

			foreach (var output in Outputs)
			{
				var outputNode = new MlpNeuron(new List<Connection>(), new List<Connection> { output }, FunctionType.Linear);
				outputLayer.Add(outputNode);
				Neurons.Add(outputNode);
			}

			return outputLayer;
		}

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
						var connection = new Connection();
						fromNeuron.Outputs.Add(connection);

						if (! (toNeuron is BiasNode))
						{
							toNeuron.Inputs.Add(connection);
						}
					}
				}
			}
		}
	}
}


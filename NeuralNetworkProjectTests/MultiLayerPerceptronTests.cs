using NUnit.Framework;
using System;
using NeuralNetworkProject;
using System.Collections.Generic;

namespace NeuralNetworkProjectTests
{
	[TestFixture]
	public class MultiLayerPerceptronTests
	{
		MultiLayerPerceptron _testNetwork;
		int _numInputs;
		int _numOutputs;
		int _numHiddenLayers;
		int _numHiddenNeuronsPerLayer;


		[SetUp]
		public void SetUp()
		{
			_numInputs = 2;
			_numOutputs = 1;
			_numHiddenLayers = 1;
			_numHiddenNeuronsPerLayer = 3;

			_testNetwork = new MultiLayerPerceptron(_numInputs, _numOutputs, _numHiddenLayers, _numHiddenNeuronsPerLayer);
		}

		[Test]
		public void TestBuildNetwork()
		{
			var inputList = _testNetwork.Inputs as List<Connection>;
			var outputList = _testNetwork.Outputs as List<Connection>;

			Assert.That(inputList.Count, Is.EqualTo(_numInputs));
			Assert.That(outputList.Count, Is.EqualTo(_numOutputs));
			Assert.That(_testNetwork.Layers.Count, Is.EqualTo(_numHiddenLayers + 2));

			var inputLayer = _testNetwork.Layers[0] as List<Neuron>;
			var hiddenLayer = _testNetwork.Layers[1] as List<Neuron>;
			var outputLayer = _testNetwork.Layers[2] as List<Neuron>;
			Assert.That(hiddenLayer.Count, Is.EqualTo(_numHiddenNeuronsPerLayer + 1)); // Add one for bias node

			var connection00 = inputLayer[0].Outputs[0];
			var connection01 = inputLayer[0].Outputs[1];
			var connection12 = hiddenLayer[0].Outputs[0];
			Assert.That(hiddenLayer[0].Inputs.Contains(connection00));
			Assert.That(hiddenLayer[1].Inputs.Contains(connection01));
			Assert.That(outputLayer[0].Inputs.Contains(connection12));
		}
	}
}


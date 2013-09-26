using NUnit.Framework;
using NeuralNetworkProject;
using System.Collections.Generic;
using System;
using System.Linq;

namespace NeuralNetworkProjectTests
{
	[TestFixture]
	public class MlpNeuronTests
	{
		MlpNeuron _testNeuron;
		Connection _testInput1;
		Connection _testInput2;
		List<Connection> _testInputs;
		Connection _testOutput;
		List<Connection> _testOutputs;
		double _sigmoidAlpha;

		[SetUp]
		public void SetUp()
		{
			_testInput1 = new Connection (1, 0.1);
			_testInput2 = new Connection (2, 0.2);
			_testInputs = new List<Connection> { _testInput1, _testInput2 };

			_testOutput = new Connection (0, 1.0);
			_testOutputs = new List<Connection> { _testOutput };

			_sigmoidAlpha = TunableParameterService.Instance.SigmoidAlpha;

			_testNeuron = new MlpNeuron (_testInputs, _testOutputs, FunctionType.Sigmoid);
		}

		[Test]
		public void TestNet()
		{
			const double expectedSum = 0.5;

			double actualSum = _testNeuron.Net();

			Assert.That (actualSum, Is.EqualTo (expectedSum));
		}

		[Test]
		public void TestLogisticFunction()
		{
			double expectedOutput = 1 / (1 + (Math.Pow(Math.E, -0.5 * _sigmoidAlpha)));

			_testNeuron.Evaluate();
			Connection actualOutputConnection = _testNeuron.Outputs.ElementAt(0);
			Assert.That (actualOutputConnection.Value, Is.EqualTo (expectedOutput));
		}
	}
}


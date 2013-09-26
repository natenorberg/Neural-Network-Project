using System;
using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public class FunctionApproximator
	{
		int _iterations;
		int _trainingSetSize;
		int _testSetSize;
		DataPoint[] _trainingSet;
		DataPoint[] _testSet;
		bool _printMovingAverage;
		int _averageWindowSize;
		double _lowerBound;
		double _upperBound;
		bool _printNetworkWeights;
		int _printWeightsFrequency;

		public FunctionApproximator() : this(TunableParameterService.Instance)
		{}

		public FunctionApproximator(ITunableParameterService paramService)
		{
			_iterations = paramService.Iterations;
			_trainingSetSize = paramService.TrainingSetSize;
			_testSetSize = paramService.TestSetSize;
			_printMovingAverage = paramService.PrintMovingAverage;
			_averageWindowSize = paramService.AverageWindowSize;
			_lowerBound = paramService.LowerBound;
			_upperBound = paramService.UpperBound;
			_printNetworkWeights = paramService.PrintNetworkWeights;
			_printWeightsFrequency = paramService.PrintWeightsFrequency;
		}

		public void ApproximateFunction(NeuralNetwork network)
		{
			Console.Write("Generating training set:\n");
			var inputs = (List<Connection>)network.Inputs;
			GenerateTrainingSet(inputs.Count);
			GenerateTestSet(inputs.Count);

			int runCount = 0;
			double totalPercentError = 0;

			while (runCount < _iterations)
			{
				foreach (var testCase in _trainingSet)
				{
					var returnValues = (List<double>)network.RunNetwork(new List<double>(testCase.Inputs));
					network.UpdateNetwork(new List<double>(testCase.Outputs));

					double expectedOut = testCase.Outputs[0];
					double actualOut = returnValues[0];

					double percentError = 100 * (Math.Abs(expectedOut - actualOut) / expectedOut);

					if (_printMovingAverage)
					{
						totalPercentError += percentError;
						if (runCount % _averageWindowSize == 0 && runCount != 0)
						{
							double averageError = totalPercentError / _averageWindowSize;
							Console.Write("Average percent error between runs {0} and {1}: {2}%\n", runCount - (_averageWindowSize - 1), runCount, averageError);
							totalPercentError = 0;
						}
					}

					if (_printNetworkWeights && runCount % _printWeightsFrequency == 0)
					{
						network.PrintWeights();
					}

					runCount++;

				}
			}
		}

		private void GenerateTrainingSet(int numInputs)
		{
			_trainingSet = GenerateDataSet(_trainingSetSize, numInputs);
		}

		private void GenerateTestSet(int numInputs)
		{
			_testSet = GenerateDataSet(_testSetSize, numInputs);
		}

		private DataPoint[] GenerateDataSet(int size, int numInputs)
		{
			DataPoint[] dataSet = new DataPoint[size];
			Random random = new Random();

			for (int i=0; i<size; i++)
			{
				var inputVector = new double[numInputs];
				var outputVector = new double[1];

				for (int j=0; j<numInputs; j++)
				{
					inputVector[j] = random.NextDouble() * (_upperBound - _lowerBound) + _lowerBound;
				}

				outputVector = RosenbrockFunction(inputVector);

				dataSet[i] = new DataPoint(inputVector, outputVector);
			}

			return dataSet;
		}

		private double[] RosenbrockFunction(double[] inputs)
		{
			double output = 0;

			for (int i=0; i<inputs.Length-1; i++)
			{
				output += Math.Pow(1 - inputs[i], 2) + 100 * Math.Pow(inputs[i + 1] - Math.Pow(inputs[i], 2), 2);
			}

			double[] outputVector = new double[1];
			outputVector[0] = output;
			return outputVector;
		}
	}

}


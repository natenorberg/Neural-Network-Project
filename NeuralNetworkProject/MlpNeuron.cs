using System.Collections.Generic;
using System;

namespace NeuralNetworkProject
{
	public class MlpNeuron : Neuron
	{
		FunctionType _functionType;
		double _output;
		double _sigmoidAlpha;

		public MlpNeuron (List<Connection> inputs, List<Connection> outputs, FunctionType functionType)
			: this (inputs, outputs, functionType, TunableParameterService.Instance)
		{}

		public MlpNeuron (List<Connection> inputs, List<Connection> outputs, 
		                  FunctionType functionType, ITunableParameterService paramService)
		{
			Inputs = inputs;
			Outputs = outputs;
			_functionType = functionType;
			_sigmoidAlpha = paramService.SigmoidAlpha;
		}

		/// <summary>
		/// Gets or sets the gradient.
		/// </summary>
		public double Gradient { get; set; }

		/// <summary>
		/// Gets the output value.
		/// </summary>
		public double OutputValue { get; private set;}

		/// <summary>
		/// Sums up all of the inputs
		/// </summary>
		public double Net()
		{
			double sum = 0;

			foreach (var input in Inputs)
			{
				sum += input.Value * input.Weight;
			}

			return sum;
		}

		/// <summary>
		/// Evaluate this neuron.
		/// </summary>
		public override void Evaluate()
		{
			double net = Net();

			_output = ActivationFunction(net);

			foreach (var output in Outputs)
			{
				output.Value = _output;
			}
		}

		private double ActivationFunction(double input)
		{
			if (_functionType == FunctionType.Sigmoid)
			{
				return SigmoidFunction(input);
			}
			else 
			{
				return input; //Linear function
			}
		}

		private double SigmoidFunction (double input)
		{
			//Logistic function : 1 / (1 + e^-x)
			return 1 / (1 + (Math.Pow(Math.E, -_sigmoidAlpha * input)));
		}
	}

	public enum FunctionType
	{
		Linear,
		Sigmoid
	}
}


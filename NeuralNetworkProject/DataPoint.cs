using System;
using System.Collections.Generic;

namespace NeuralNetworkProject
{
	public class DataPoint
	{
		public DataPoint(double[] inputs, double[] outputs)
		{
			Inputs = inputs;
			Outputs = outputs;
		}

		/// <summary>
		/// Gets or sets the inputs.
		/// </summary>
		public double[] Inputs { get; private set; }

		/// <summary>
		/// Gets or sets the outputs.
		/// </summary>
		public double[] Outputs { get; private set; }
	}
}


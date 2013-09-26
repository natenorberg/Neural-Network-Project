namespace NeuralNetworkProject
{
	public interface ITunableParameterService
	{
		/// <summary>
		/// The scalar used in sigmoid functions
		/// </summary>
		double SigmoidAlpha { get; }

		/// <summary>
		/// Whether or not to use momentum
		/// </summary>
		bool IsMomentumUsed { get; }

		/// <summary>
		/// Returns the constant used for momentum
		/// </summary>
		double MomentumAmount { get; }

		/// <summary>
		/// Gets the learning rate.
		/// </summary>
		double LearningRate { get; }

		/// <summary>
		/// Gets the number of iterations.
		/// </summary>
		int Iterations { get; }

		/// <summary>
		/// Gets the size of the training set.
		/// </summary>
		int TrainingSetSize { get; }

		/// <summary>
		/// Gets the size of the test set.
		/// </summary>
		int TestSetSize { get; }

		/// <summary>
		/// Whether the output should print the moving average
		/// </summary>
		bool PrintMovingAverage { get; }

		/// <summary>
		/// Gets the window for average size.
		/// </summary>
		int AverageWindowSize { get; }

		/// <summary>
		/// Gets the lower bound.
		/// </summary>
		double LowerBound { get; }

		/// <summary>
		/// Gets the upper bound.
		/// </summary>
		double UpperBound { get; }

		/// <summary>
		/// Gets the type of the neural network.
		/// </summary>
		NetworkType NeuralNetworkType {get; }

		/// <summary>
		/// Gets the number of inputs.
		/// </summary>
		int NumberOfInputs { get; }

		/// <summary>
		/// Gets the number of outputs.
		/// </summary>
		int NumberOfOutputs { get; }

		/// <summary>
		/// Gets the number of hidden layers.
		/// </summary>
		int NumberOfHiddenLayers { get; }

		/// <summary>
		/// Gets the number of nodes per hidden layer.
		/// </summary>
		int NumberOfNodesPerHiddenLayer { get; }

		/// <summary>
		/// Gets the number of centers.
		/// </summary>
		int NumberOfCenters { get; }

		/// <summary>
		/// Whether or not the output should show network weight information
		/// </summary>
		bool PrintNetworkWeights { get; }

		/// <summary>
		/// How often the weight information should be printed
		/// </summary>
		int PrintWeightsFrequency { get; }

		/// <summary>
		/// Gets a value indicating whether to use annealling.
		/// </summary>
		bool IsAnnealingUsed{ get; }

		/// <summary>
		/// Gets the annealing value.
		/// </summary>
		double AnnealingValue{ get; }
	}
}


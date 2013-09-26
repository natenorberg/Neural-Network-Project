namespace NeuralNetworkProject
{
	public class TunableParameterService : ITunableParameterService
	{
		private static TunableParameterService _instance;

		TunableParameterService ()
		{
			// Network Creation
			NeuralNetworkType = NetworkType.RBF;
			NumberOfInputs = 3;
			NumberOfOutputs = 1;
			// MLP
			NumberOfHiddenLayers = 1;
			NumberOfNodesPerHiddenLayer = 4;
			// RBF
			NumberOfCenters = 8; // Should be a square number for centers to be even

			// General Parameters
			LearningRate = 0.05;
			IsAnnealingUsed = false;
			AnnealingValue = 0.000001; // Needs to be very small
			IsMomentumUsed = true;
			MomentumAmount = 0.1;
			Iterations = int.MaxValue;
			TrainingSetSize = 10000000;
			TestSetSize = 1000;
			LowerBound = -0.1;
			UpperBound = 0.1;

			// Multi Layer Perceptron Parameters
			SigmoidAlpha = 0.5;

			// Print parameters
			PrintMovingAverage = true;
			AverageWindowSize = 10000;
			PrintNetworkWeights = false;
			PrintWeightsFrequency = 100;
		}

		/// <summary>
		/// Returns the singleton instance
		/// </summary>
		public static TunableParameterService Instance
		{ 
			get {
				if (_instance == null) {
					_instance = new TunableParameterService();
				}

				return _instance;
			}
		}

		/// <summary>
		/// The scalar used in sigmoid functions
		/// </summary>
		public double SigmoidAlpha { get; private set; }

		/// <summary>
		/// Whether or not to use momentum
		/// </summary>
		public bool IsMomentumUsed { get; private set; }

		/// <summary>
		/// Returns the constant used for momentum
		/// </summary>
		public double MomentumAmount { get; private set; }

		/// <summary>
		/// Gets the learning rate.
		/// </summary>
		public double LearningRate { get; private set; }

		/// <summary>
		/// Gets the number of iterations.
		/// </summary>
		public int Iterations { get; private set;}

		/// <summary>
		/// Gets the size of the training set.
		/// </summary>
		public int TrainingSetSize { get; private set; }

		/// <summary>
		/// Gets the size of the test set.
		/// </summary>
		public int TestSetSize { get; private set; }

		/// <summary>
		/// Whether the output should print the moving average
		/// </summary>
		public bool PrintMovingAverage { get; private set; }

		/// <summary>
		/// Gets the window for average size.
		/// </summary>
		public int AverageWindowSize { get; private set; }

		/// <summary>
		/// Gets the lower bound.
		/// </summary>
		public double LowerBound { get; private set;}

		/// <summary>
		/// Gets the upper bound.
		/// </summary>
		public double UpperBound { get; private set; }

		/// <summary>
		/// Gets or sets the type of the neural network.
		/// </summary>
		public NetworkType NeuralNetworkType { get; private set; }

		/// <summary>
		/// Gets the number of inputs.
		/// </summary>
		public int NumberOfInputs { get; private set; }

		/// <summary>
		/// Gets the number of outputs.
		/// </summary>
		public int NumberOfOutputs { get; private set; }

		/// <summary>
		/// Gets the number of hidden layers.
		/// </summary>
		public int NumberOfHiddenLayers { get; private set; }

		/// <summary>
		/// Gets the number of nodes per hidden layer.
		/// </summary>
		public int NumberOfNodesPerHiddenLayer { get; private set; }

		/// <summary>
		/// Gets the number of centers.
		/// </summary>
		public int NumberOfCenters { get; private set; }

		/// <summary>
		/// Whether or not the output should show network weight information
		/// </summary>
		public bool PrintNetworkWeights { get; private set; }

		/// <summary>
		/// How often the weight information should be printed
		/// </summary>
		public int PrintWeightsFrequency { get; private set; }

		/// <summary>
		/// Gets a value indicating whether to use annealling.
		/// </summary>
		public bool IsAnnealingUsed{ get; private set; }

		/// <summary>
		/// Gets the annealing value.
		/// </summary>
		public double AnnealingValue{ get; private set;}
	}
}


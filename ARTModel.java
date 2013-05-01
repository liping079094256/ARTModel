package cc.mallet.topics;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.types.*;
import cc.mallet.classify.MaxEnt;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.Noop;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.io.File;

/**
 * This code is an implementation of Author & reference topic model!
 * 
 * 
 * 
 * ***/
public class ARTModel extends LDAHyper
{

	MaxEnt dmrParameters = null;
	int numFeatures;
	int defaultFeatureIndex;

	Pipe parameterPipe = null;

	double[][] alphaCache;
	double[] alphaSumCache;
	BufferedWriter bw;

	public ARTModel(int numberOfTopics,BufferedWriter bw)
	{
		super(numberOfTopics);
		this.bw=bw;
	}

	
	/**
	 * EM step of ARTModel!
	 * ***/
	public void estimate(int iterationsThisRound) throws IOException
	{

		numFeatures = data.get(0).instance.getTargetAlphabet().size() + 1;
		defaultFeatureIndex = numFeatures - 1;

		int numDocs = data.size(); // 

		alphaCache = new double[numDocs][numTopics];
		alphaSumCache = new double[numDocs];

		long startTime = System.currentTimeMillis();
		int maxIteration = iterationsSoFar + iterationsThisRound;

		for (; iterationsSoFar <= maxIteration; iterationsSoFar++)
		{
			long iterationStart = System.currentTimeMillis();

			if (showTopicsInterval != 0 && iterationsSoFar != 0
					&& iterationsSoFar % showTopicsInterval == 0)
			{
				System.out.println();
				printTopWords(System.out, wordsPerTopic, false);
			}

			if (saveStateInterval != 0
					&& iterationsSoFar % saveStateInterval == 0)
			{
				this.printState(new File(stateFilename + '.' + iterationsSoFar
						+ ".gz"));
			}

			if (iterationsSoFar > burninPeriod && optimizeInterval != 0
					&& iterationsSoFar % optimizeInterval == 0)
			{

				// Train regression parameters
				learnParameters();
			}

			// Loop over every document in the corpus

			for (int doc = 0; doc < numDocs; doc++)
			{
				FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).instance
						.getData();
				LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;

				if (dmrParameters != null)
				{
					// set appropriate Alpha parameters
					setAlphas(data.get(doc).instance);
				}

				sampleTopicsForOneDoc(tokenSequence, topicSequence, false,
						false);
			}

			long ms = System.currentTimeMillis() - iterationStart;
			if (ms > 1000)
			{
				System.out.print(Math.round(ms / 1000) + "s ");
			}
			else
			{
				System.out.print(ms + "ms ");
			}

			if (iterationsSoFar % 10 == 0)
			{
				System.out.println("<" + iterationsSoFar + "> ");
				bw.append(modelLogLikelihood()+"\n");
				bw.flush();
				if (printLogLikelihood)
					System.out.println(modelLogLikelihood());
			}
			System.out.flush();
		}

		long seconds = Math.round((System.currentTimeMillis() - startTime) / 1000.0);
		long minutes = seconds / 60;
		seconds %= 60;
		long hours = minutes / 60;
		minutes %= 60;
		long days = hours / 24;
		hours %= 24;
		System.out.print("\nTotal time: ");
		if (days != 0)
		{
			System.out.print(days);
			System.out.print(" days ");
		}
		if (hours != 0)
		{
			System.out.print(hours);
			System.out.print(" hours ");
		}
		if (minutes != 0)
		{
			System.out.print(minutes);
			System.out.print(" minutes ");
		}
		System.out.print(seconds);
		System.out.println(" seconds");
	}

	public void setAlphas()
	{

		double[] parameters = dmrParameters.getParameters();

		alphaSum = 0.0;
		smoothingOnlyMass = 0.0;
		for (int topic = 0; topic < numTopics; topic++)
		{
			alpha[topic] = Math.exp(parameters[(topic * numFeatures)
					+ defaultFeatureIndex]);
			alphaSum += alpha[topic];

			smoothingOnlyMass += alpha[topic] * beta
					/ (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] = alpha[topic]
					/ (tokensPerTopic[topic] + betaSum);
		}

	}

	public void setAlphas(int featureIndex)
	{

		double[] parameters = dmrParameters.getParameters();

		alphaSum = 0.0;
		smoothingOnlyMass = 0.0;
		for (int topic = 0; topic < numTopics; topic++)
		{
			alpha[topic] = Math.exp(parameters[(topic * numFeatures)
					+ featureIndex]
					+ parameters[(topic * numFeatures) + defaultFeatureIndex]);
			alphaSum += alpha[topic];

			smoothingOnlyMass += alpha[topic] * beta
					/ (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] = alpha[topic]
					/ (tokensPerTopic[topic] + betaSum);
		}

	}

	/**
	 * Set alpha based on features in an instance
	 */
	public void setAlphas(Instance instance)
	{

		// we can't use the standard score functions from MaxEnt,
		// since our features are currently in the Target.
		FeatureVector features = (FeatureVector) instance.getTarget();
		if (features == null)
		{
			setAlphas();
			return;
		}

		double[] parameters = dmrParameters.getParameters();

		alphaSum = 0.0;
		smoothingOnlyMass = 0.0;

		for (int topic = 0; topic < numTopics; topic++)
		{
			alpha[topic] = parameters[topic * numFeatures + defaultFeatureIndex]
					+ MatrixOps.rowDotProduct(parameters, numFeatures, topic,
							features, defaultFeatureIndex, null);

			alpha[topic] = Math.exp(alpha[topic]);
			alphaSum += alpha[topic];

			smoothingOnlyMass += alpha[topic] * beta
					/ (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] = alpha[topic]
					/ (tokensPerTopic[topic] + betaSum);
		}
	}

	public void learnParameters()
	{
		if (parameterPipe == null)
		{
			parameterPipe = new Noop();

			parameterPipe.setDataAlphabet(data.get(0).instance
					.getTargetAlphabet());
			parameterPipe.setTargetAlphabet(topicAlphabet);
		}

		InstanceList parameterInstances = new InstanceList(parameterPipe);

		if (dmrParameters == null)
		{
			dmrParameters = new MaxEnt(parameterPipe, new double[numFeatures
					* numTopics]);
		}

		for (int doc = 0; doc < data.size(); doc++)
		{

			if (data.get(doc).instance.getTarget() == null)
			{
				continue;
			}

			FeatureCounter counter = new FeatureCounter(topicAlphabet);

			for (int topic : data.get(doc).topicSequence.getFeatures())
			{
				counter.increment(topic);
			}

			parameterInstances.add(new Instance(data.get(doc).instance
					.getTarget(), counter.toFeatureVector(), null, null));

		}

		ARTOptimizable optimizable = new ARTOptimizable(parameterInstances,
				dmrParameters);
		optimizable.setRegularGaussianPriorVariance(0.5);
		optimizable.setInterceptGaussianPriorVariance(100.0);

		LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

		try
		{
			optimizer.optimize();
		}
		catch (OptimizationException e)
		{
			
		}

		try
		{
			optimizer.optimize();
		}
		catch (OptimizationException e)
		{
		
		}
		dmrParameters = optimizable.getClassifier();

		for (int doc = 0; doc < data.size(); doc++)
		{
			Instance instance = data.get(doc).instance;
			FeatureSequence tokens = (FeatureSequence) instance.getData();
			if (instance.getTarget() == null)
			{
				continue;
			}
			int numTokens = tokens.getLength();

			setAlphas(instance);

			for (int topic = 0; topic < numTopics; topic++)
			{
				alphaCache[doc][topic] = alpha[topic];
			}
			alphaSumCache[doc] = alphaSum;
		}
	}

	public void printTopWords(PrintStream out, int numWords,
			boolean usingNewLines)
	{
		if (dmrParameters != null)
		{
			setAlphas();
		}
		super.printTopWords(out, numWords, usingNewLines);
	}

	public void writeParameters(File parameterFile) throws IOException
	{
		if (dmrParameters != null)
		{
			PrintStream out = new PrintStream(parameterFile);
			dmrParameters.print(out);
			out.close();
		}
	}

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;

	public static void main(String[] args) throws IOException
	{
		ObjectInputStream in=null;
		InstanceList list=null;
		try{
			in=new ObjectInputStream(new FileInputStream(new File("C:\\Documents and Settings\\Administrator\\桌面\\DMR实验数据\\c.txt")));
			list=(InstanceList) in.readObject();
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		

		int numTopics = args.length > 1 ? Integer.parseInt(args[1]) : 200;

		InstanceList testing = args.length > 2 ? InstanceList.load(new File(
				args[2])) : null;

		BufferedWriter bw=new BufferedWriter(new FileWriter(new File("c:\\DMRlikelihood.txt")));
		ARTModel lda = new ARTModel(numTopics,bw);
		lda.setOptimizeInterval(100);
		lda.setTopicDisplay(100, 10);
		lda.addInstances(list);
		lda.estimate();
		lda.writeParameters(new File("dmr.parameters"));
		lda.printState(new File("dmr.state.gz"));
		
		
	}
}
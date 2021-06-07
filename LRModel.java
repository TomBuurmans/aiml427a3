import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import scala.Tuple2;

/**
 * Logistic regression based classification using ML Lib.
 */
public final class LRModel {
	
	static final long SPLIT_SEED = new Random().nextLong();
	
	private void init(String inputPath, String outputPath) {
		long startTime = System.currentTimeMillis();

		//configure spark
		SparkConf conf = new SparkConf().setAppName("LogisticRegression").setMaster("yarn-cluster");
	    SparkContext sc = new SparkContext(conf);
	    JavaSparkContext  context = new JavaSparkContext(sc);

	    // Load dataset
		RDD<String> input = sc.textFile(inputPath, 1);
	    JavaRDD<String> data = input.toJavaRDD(); 
	    
	    //split dataset into training and test set with random seed
		JavaRDD<String>[] split = data.randomSplit(new double[] {0.7, 0.3},SPLIT_SEED);
		
		JavaRDD<String> train = split[0].cache();
		JavaRDD<String> test  = split[1];
		
		//returns the dataset class into binary form
		JavaRDD<LabeledPoint> trainLabeledPoints = getLabeledPoints(train);
		JavaRDD<LabeledPoint> testLabeledPoints = getLabeledPoints(test);

	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TRAINING >>>>>>>>>>>>>>>>>>>>>>>>");

	    // Train Logistic Regression model
	    org.apache.spark.mllib.classification.LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
	    	      .setNumClasses(2)
	    	      .setIntercept(true)
	    	      .run(trainLabeledPoints.rdd());  
	    
	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TESTING >>>>>>>>>>>>>>>>>>>>>>>>");
	    
	  //Testing the model
	  		JavaPairRDD<Object, Object> predictionAndLabelsTrain = trainLabeledPoints.mapToPair(p ->
	          new Tuple2<>(model.predict(p.features()), p.label()));

	  	    JavaPairRDD<Object, Object> predictionAndLabelsTest = testLabeledPoints.mapToPair(p ->
	          new Tuple2<>(model.predict(p.features()), p.label()));
	  	    
	  	    //calculates the training accuracy
	  		MulticlassMetrics metricsTrain = new MulticlassMetrics(predictionAndLabelsTrain.rdd());
	  	    double trainAccuracy = metricsTrain.accuracy();
	  	    System.out.println("Train Accuracy = " + trainAccuracy);
	  	    
	  	    double trainErr = 1.0 - trainAccuracy;
	  	    System.out.println("Train Error = "+ trainErr );

	  	    //calculates the test accuracy
	  	    MulticlassMetrics metricsTest = new MulticlassMetrics(predictionAndLabelsTest.rdd());
	  	    double testAccuracy = metricsTest.accuracy();
	  	    System.out.println("Test Accuracy = " + testAccuracy);
	  	    
	  	    double testErr = 1.0 - testAccuracy;
	  	    System.out.println("Test Error = "+ testErr );

	  		System.out.print("Final w: " + model.weights());
	  		System.out.println("LR Model\n" +model.toString());
	  		
	  		long timeTaken = (System.currentTimeMillis() - startTime) / 1000;
	  		
	  		 String outputString = "Train Accuracy: " + trainAccuracy + "\n";
	 		outputString += "Train Error: " + trainErr + "\n";
	 		outputString += "Test Accuracy: " + testAccuracy + "\n";
	 		outputString += "Test Error: " + testErr + "\n";
	 		outputString += "Duration: " + timeTaken + "ms\n";
	 		outputString += "Final weights: "+ model.weights();
	 		outputString += "Logistic Regression model: "+ model.toString();

	         List<String> outputStrings = new ArrayList<String>();
	 		outputStrings.add(outputString);
	 		
	 		JavaRDD<String> output = context.parallelize(outputStrings);
	 		output.saveAsTextFile(outputPath);

	    sc.stop();
		
	}	
	
	
	private JavaRDD<LabeledPoint> getLabeledPoints(JavaRDD<String> train) {
		return train.map(p -> {
			String[] tokens = p.split(",");
			double[] features = new double[tokens.length - 1];
			for (int i = 0; i < features.length; i++) {
				features[i] = Double.parseDouble(tokens[i]);
			}
			DenseVector v = new DenseVector(features);
			if (tokens[features.length].equals("normal")) {
				return new LabeledPoint(0.0, v);
			} else {
				return new LabeledPoint(1.0, v);
			}
		});
	}

	
	public static void main(String[] args) {

		LRModel runLR = new LRModel();
		runLR.init(args[0],args[1]);
	}

}
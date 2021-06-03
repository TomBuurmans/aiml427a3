import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
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
	
	private void init(String[] args) {
		
		//configure spark
		SparkConf conf = new SparkConf().setAppName("LogisticRegression").setMaster("local");
	    SparkContext sc = new SparkContext(conf);

	    // Load dataset
		RDD<String> input = sc.textFile("kdd.data", 1);
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
	    	      .setNumClasses(10)
	    	      .run(trainLabeledPoints.rdd());  
	    
	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TRAINING >>>>>>>>>>>>>>>>>>>>>>>>");
	    
	    //Testing the model
	    JavaPairRDD<Object, Object> predictionAndLabels = testLabeledPoints.mapToPair(p ->
        new Tuple2<>(model.predict(p.features()), p.label()));
	    
	    
	    //calculates the accuracy
	    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
	    double accuracy = metrics.accuracy();
	    System.out.println("Accuracy = " + accuracy);
	    
	    double testErr = 1.0 - accuracy;
	    System.out.println("Test Error = "+ testErr );

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
		runLR.init(args);
	}

}
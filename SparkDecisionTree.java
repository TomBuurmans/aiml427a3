import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.ml.feature.StandardScaler;
import scala.Tuple2;

import static org.apache.spark.sql.functions.*;

public class SparkDecisionTree implements Serializable {
    
    private JavaSparkContext context;
    
    public static void main(String[] args) {
		new SparkDecisionTree().run(args[0], args[1]);
	}
    
    public void run(String dataDir, String outputDir) {
		long start = System.currentTimeMillis();

		SparkSession spark = SparkSession.builder()
			.appName("Spark Decision Tree")
			.getOrCreate();
		context = new JavaSparkContext(spark.sparkContext());

		Dataset<Row> kddData = spark.read().schema(buildSchema(42)).csv(dataDir);
		Dataset<Row> indexed = new StringIndexer()
			.setInputCol("_c41")
			.setOutputCol("connection_type")
			.fit(kddData)
			.transform(kddData)
			.drop("_c41");

		Dataset<Row>[] split = indexed.randomSplit(new double[] {0.7, 0.3});
		JavaRDD<Row> train = split[0].toJavaRDD();
		JavaRDD<Row> test  = split[1].toJavaRDD();

		JavaRDD<LabeledPoint> trainLabeledPoints = getLabeledPoints(train);
		JavaRDD<LabeledPoint> testLabeledPoints = getLabeledPoints(test);

		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		Integer maxDepth = 16;// try others
		Integer maxBins = 64;// try others
		String impurity = "gini"// try entropy but in theory gini should be better
		final DecisionTreeModel model = DecisionTree.trainClassifier(trainLabeledPoints, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

		JavaPairRDD<Double, Double> train_yHatToY = trainLabeledPoints.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		double trainErr = train_yHatToY.filter(pl -> !pl._1().equals(pl._2())).count() / (double) trainLabeledPoints.count();
		double train_accuracy = 1 - trainErr;

		JavaPairRDD<Double, Double> yHatToY = testLabeledPoints.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		double testErr = yHatToY.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testLabeledPoints.count();
		double accuracy = 1 - testErr;

		long duration = (System.currentTimeMillis() - start) / 1000;

        String outputString = "Train Accuracy: " + train_accuracy + "\n";
		outputString += "Train Error: " + trainErr + "\n";
		outputString += "Test Accuracy: " + accuracy + "\n";
		outputString += "Test Error: " + testErr + "\n";
		outputString += "Duration: " + duration + "ms\n";
		outputString += "Tree model:\n" + model.toDebugString() + "\n";

        JavaRDD<String> output = context.parallelize(new ArrayList<String>() {{
            add(outputString);
        }});
		output.saveAsTextFile(outputDir);
	}
    
    private StructType buildSchema(int l) {
		StructField[] f = new StructField[l];
		for(int i = 0; i < l-1; i++) {
			f[i] = DataTypes.createStructField("_c" + i, DataTypes.DoubleType, true);
		}
		f[l-1] = DataTypes.createStructField("_c" + (l-1), DataTypes.StringType, true);
		return new StructType(f);
	}
    
    private JavaRDD<LabeledPoint> getLabeledPoints(JavaRDD<Row> set) {
		return set.map(p -> {
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
}
package weka.custom_classifier;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;

import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

public class Main {
	private static final String dataSet = "example/weather.numeric.arff";
	
	
	public static Instances loadDatasetArff(String filePath) throws IOException
    { 
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filePath));
		return loader.getDataSet();
    }
	
	public static void main(String[] args) throws Exception{
		Id3 decisionTree = new Id3();
		
		Instances data = Main.loadDatasetArff(dataSet);
		data.setClass(data.attribute(data.numAttributes() - 1));
		decisionTree.buildClassifier(data);
		
	}
}

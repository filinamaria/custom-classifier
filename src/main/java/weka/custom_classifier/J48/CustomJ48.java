package weka.custom_classifier.J48;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Stack;

import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import com.google.common.math.DoubleMath;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.custom_classifier.Tree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class CustomJ48 extends Classifier{
	/** For serialization */
	private static final long serialVersionUID = 1L;
	
	private Tree decisionTree;
	private double confidenceFactor;
	private ArrayList<Integer> infoBinarySplit;
	private int[] numOfMissingValues;
	private ArrayList <ArrayList<Integer>> largestAttributeDistribution;
	 
	public CustomJ48(){
		decisionTree = new Tree();
		confidenceFactor = 0.25f;
		infoBinarySplit = new ArrayList();
		largestAttributeDistribution = new ArrayList();
	}
	
	/**
	 * @return
	 */
	private Capabilities classifierCapabilities(){
		Capabilities capabilities = super.getCapabilities();
		
		// attributes
		capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
		capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
		capabilities.enable(Capability.MISSING_VALUES);

	    // class
		capabilities.enable(Capability.NOMINAL_CLASS);

	    // instances
		capabilities.setMinimumNumberInstances(0);
		
		return capabilities;
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		classifierCapabilities().testWithFail(data);
		
		data.deleteWithMissingClass();
		
		numOfMissingValues = new int[data.numAttributes()];
		
		data = new Instances(splitNumericAttribute(data));
		generateAttributesValueDistribution(data);
		data = new Instances(replaceMissingAttribute(data));
		System.out.println(data);
		System.out.println("Generating Tree");
		generateTree(data, decisionTree);
	}
	
	private Instances splitNumericAttribute(Instances data) throws Exception
	{
		Instances retVal = new Instances(data);
		
		int len = data.numAttributes();
		int count0=0;
		int count1=0;
		for (int i=0; i<len; i++)
		{
			if (data.attribute(i).isNumeric()) {
				retVal.deleteAttributeAt(i);
				
				double valMin = Double.MAX_VALUE;
				double valMax = Double.MIN_VALUE;
				for (int j=0; j<data.numInstances(); j++)
				{
					if (data.instance(j).value(i) > valMax)
					{
						valMax = data.instance(j).value(i);
					}
					if (data.instance(j).value(i) < valMin)
					{
						valMin = data.instance(j).value(i);
					}
				}
				int middleValue = (int) Math.floor((valMax-valMin)/2 + valMin);
				infoBinarySplit.add(middleValue);
				
				Add filter = new Add();
				System.out.println("flag: " + i);
				filter.setAttributeIndex("first");
				filter.setNominalLabels("moreThan"+Integer.toString(middleValue)+", lessEqualThan"+Double.toString(middleValue));
				filter.setAttributeName(data.attribute(i).name());
				filter.setInputFormat(retVal);
				retVal = Filter.useFilter(retVal, filter);
				
				for (int j=0; j<data.numInstances(); j++)
				{
					if (!data.instance(j).isMissing(i))
					{		
						if (Double.compare(data.instance(j).value(i), middleValue) > 0) {
							retVal.instance(j).setValue(retVal.attribute(data.attribute(i).name()), 0);
							count0++;
						}
						else if (Double.compare(data.instance(j).value(i), middleValue) <= 0)  {
							retVal.instance(j).setValue(retVal.attribute(data.attribute(i).name()), 1);
							count1++;
						}
					}
					//penanganan missing value pada fungsi berikutnya
				}
			}
		}
		return retVal;
	}
	
	public void generateAttributesValueDistribution(Instances data)
	{
		int numAttr = data.numAttributes();
		int maxAttr = Integer.MIN_VALUE;
		int simpan = 0;
		for (int i=0; i<numAttr; i++)
			if (data.attribute(i).numValues() > maxAttr) {
				maxAttr = data.attribute(i).numValues();
				simpan = i;
			}
		int[][][] counter = new int[data.numClasses()][data.numAttributes()][data.attribute(simpan).numValues()];
		//System.out.println(data.numClasses()+" "+data.numAttributes()+" "+data.attribute(simpan).numValues());
		
		//Counting matrix of largest attribute value distribution over classes
		for (int i=0; i<data.numInstances(); i++)
		{
			for (int j=0; j<data.numAttributes(); j++) {
				if (!data.instance(i).isMissing(j)) {
					//System.out.println(data.instance(i).classValue()+" "+j+" "+data.instance(i).value(data.instance(i).attribute(j)));
					counter[(int) data.instance(i).classValue()][j][(int) data.instance(i).value(data.instance(i).attribute(j))]++;
				}
			}
		}
		
		//Generating matrix of largest attribute value distribution over classes
		for (int i=0; i<data.numClasses(); i++) {
			ArrayList<Integer> maxAttributeIdx = new ArrayList<Integer>();
			for (int j=0; j<data.numAttributes(); j++) {
				int max = Integer.MIN_VALUE;
				simpan = 0;
				for (int k=0; k<maxAttr; k++)
					if (counter[i][j][k] > max) {
						max = (int) counter[i][j][k];
						simpan = k;
					}
				maxAttributeIdx.add(simpan);
			}
			largestAttributeDistribution.add(maxAttributeIdx);
		}		
	}
	
	/**
	 * Replace instances with missing value
	 * @param data Instances
	 * @return Instances with missing value replaced
	 */
	public Instances replaceMissingAttribute(Instances data)
	{
		Instances retVal = new Instances(data);
		for (int i=0; i<data.numInstances(); i++) 
		{
			for (int j=0; j<data.instance(i).numAttributes(); j++) {
				if (data.instance(i).isMissing(j))
				{
					retVal.instance(i).setValue(j, largestAttributeDistribution.get(data.instance(i).classIndex()).get(j));
				}
			}
		}
		return retVal;
	}
	
	public Instances splitNumericSuppliedTest(Instances data) throws Exception
	{
		Instances retVal = new Instances(data);
		int count = 0;
		int len = data.numAttributes();
		for (int i=0; i<len; i++)
		{
			if (data.attribute(i).isNumeric()) {			
				retVal.deleteAttributeAt(i);
				
				Add filter = new Add();
				filter.setAttributeIndex("first");
				filter.setNominalLabels("moreThan"+Integer.toString(infoBinarySplit.get(count))+", lessEqualThan"+Double.toString(infoBinarySplit.get(count)));
				filter.setAttributeName(data.attribute(i).name());
				filter.setInputFormat(retVal);
				retVal = Filter.useFilter(retVal, filter);
				
				for (int j=0; j<data.numInstances(); j++)
				{
					if (Double.compare(data.instance(j).value(i), infoBinarySplit.get(count)) > 0) {
						retVal.instance(j).setValue(retVal.attribute(data.attribute(i).name()), 0);
					}
					else {
						retVal.instance(j).setValue(retVal.attribute(data.attribute(i).name()), 1);
					}
				}
				count+=1;
			}
		}
		return retVal;
	}
	
	//prune
	public void prune (Tree tree){
		Stack<Integer> stack;
		stack = new Stack<Integer>();
		
		while(!tree.getAttribute()==null)
		{
			stack.push()
		}
		
		
	}
	
	
	public double classifyInstance(Instance instance){
		return classifyInstance(instance, decisionTree);
	}
	
	private double classifyInstance(Instance instance, Tree tree){
		if(tree.getAttribute() == null){
			return tree.getClassValue();
		}else{
			return classifyInstance(instance, tree.getChild((int) instance.value(tree.getAttribute())));
		}
	}
	
	/**
	 * @param data
	 * @param tree
	 */
	public void generateTree(Instances data, Tree tree){
		if(data.numInstances() == 0){
			tree.setAttribute(null);
			tree.setClassValue(Instance.missingValue());
			return;
		}
		
		Enumeration attributes = data.enumerateAttributes();
		double[] infoGains = new double[data.numAttributes()];
	
		while(attributes.hasMoreElements()){
			Attribute attribute = (Attribute) attributes.nextElement();
			double infoGain = informationGain(data, attribute);
			double splitInfo = splitInfo(data);
			if (Double.compare(splitInfo, 0.0)!=0) {
				infoGains[attribute.index()] = (infoGain / splitInfo);
			}
			else
				infoGains[attribute.index()] =infoGain;
			System.out.println("information gain: " + infoGain);
			System.out.println("split Info: " + splitInfo);
		}
		
		Attribute highestIGAtt = data.attribute(maxIndex(infoGains));
		
		// Build decision tree
		tree.setAttribute(highestIGAtt);
		
		if(Double.compare(infoGains[highestIGAtt.index()], 0.0) == 0){
			tree.setAttribute(null);
			double[] distribution = new double[data.numClasses()];
			Enumeration instances = data.enumerateInstances();
			
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
				distribution[(int) instance.classValue()]++;
			}
			
			tree.setClassValue(maxIndex(distribution));
			tree.setClassAttribute(data.classAttribute());
		}else{
			Instances[] splittedData = split(data, highestIGAtt);
			Tree[] children = new Tree[tree.getAttribute().numValues()];
			
			for(int i = 0; i < children.length; i++){
				children[i] = new Tree();
				tree.addChildren(children);
				generateTree(splittedData[i], children[i]);
			}
		}
	}
	
	private boolean isClassified(Instances data, Attribute att){
		Enumeration instances = data.enumerateInstances();
		boolean classified = true;
		Instance comparator = (Instance) instances.nextElement();
		
		while(instances.hasMoreElements() && classified){
			Instance temp = (Instance) instances.nextElement();

			if(temp.classValue() != comparator.classValue()){
				classified = false;
			}
			
			comparator = temp;
		}
		
		return classified;
	}
	
	/**
	 * @param data
	 * @param att
	 * @return
	 */
	private Instances[] split(Instances data, Attribute att){
		Instances[] splittedData = new Instances[att.numValues()];
		
		for(int i = 0; i < splittedData.length; i++){
			splittedData[i] = new Instances(data, data.numInstances());
			splittedData[i].delete();
		}
		
		for(int i = 0; i < data.numInstances(); i++){			
			splittedData[(int) data.instance(i).value(att)].add(data.instance(i));
		}
		
		return splittedData;
	}
	
	/**
	 * @param array
	 * @return
	 */
	private int maxIndex(double[] array){
		int maxIndex = 0;
		
		for (int i = 1; i < array.length; i++){
			double newnumber = array[i];
			if ((newnumber > array[maxIndex])){
				maxIndex = i;
			}
		}
		
		return maxIndex;
	}
	
	/**
	 * @param data
	 * @param att
	 * @return
	 */
	private double informationGain(Instances data, Attribute att){
		double informationGain = entropy(data);
		int numOfLabels = att.numValues();

		Instances[] instancesDistribution = new Instances[numOfLabels];
		
		for(int i = 0; i < instancesDistribution.length; i++){
			instancesDistribution[i] = new Instances(data, data.numInstances());
			instancesDistribution[i].delete();
		}
		
		for(int i = 0; i < data.numInstances(); i++){			
			instancesDistribution[(int) data.instance(i).value(att)].add(data.instance(i));
		}
		
		for(int i = 0; i < numOfLabels; i++){
			double numInstancesOfLabel = (double) instancesDistribution[i].numInstances();
			informationGain -=  numInstancesOfLabel / (double) data.numInstances() * entropy(instancesDistribution[i]);
		}
		
		return informationGain;
	}
	
	private double splitInfo(Instances data){
		double splitInfo = 0.0;
		int[]  distribution = new int[data.numClasses()];
		
		for(int i = 0; i < data.numInstances(); i++){
			distribution[(int) data.instance(i).classValue()]++;
		}
		
		for(int i = 0; i < data.numClasses(); i++){
			double temp = (double) distribution[i] / (double) data.numInstances(); 
			
			if(Double.compare(temp, 0) != 0)
				splitInfo += (temp * DoubleMath.log2(temp));
		}
		
		return Math.abs(splitInfo);
	}
	
	/**
	 * @param data
	 * @return
	 */
	private double entropy(Instances data){
		double entropy = 0.0;
		int numOfClasses = data.classAttribute().numValues();
		int[] numOfInstancesPerClass = new int[numOfClasses];
		
		for(int i = 0; i < data.numInstances(); i++){
			numOfInstancesPerClass[(int) data.instance(i).classValue()]++;
		}
		
		for(int i = 0; i < numOfClasses; i++){
			if(numOfInstancesPerClass[i] != 0){
				double temp = (double) numOfInstancesPerClass[i] / (double) data.numInstances();
				entropy -= temp * DoubleMath.log2(temp);
			}
		}
		
		return entropy;
	}
	
	private String toString(int level, Tree tree) {

	    StringBuffer text = new StringBuffer();
	    
	    if (tree.getAttribute() == null) {
	      if (Instance.isMissingValue(tree.getClassValue())) {
	        text.append(": null");
	      } else {
	        text.append(": " + tree.getClassAttribute().value((int) tree.getClassValue()));
	      } 
	    } else {
	      for (int j = 0; j < tree.getAttribute().numValues(); j++) {
	        text.append("\n");
	        for (int i = 0; i < level; i++) {
	          text.append("|  ");
	        }
	        text.append(tree.getAttribute().name() + " = " + tree.getAttribute().value(j));
	        text.append(toString(level + 1, tree.getChild(j)));
	      }
	    }
	    return text.toString();
	}
	
	public String toString() {

	    if ((decisionTree.getAttribute() == null) && (decisionTree.getChildren() == null)) {
	      return "J48: No model built yet.";
	    }
	    return "J48\n\n" + toString(0, decisionTree);
	  }
	
	public static Instances loadDatasetArff(String filePath) throws IOException
    { 
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filePath));
		return loader.getDataSet();
    }
	
	public static void main(String[] args) throws Exception{
		String dataset = "example/iris.arff";
		CustomJ48 j48 = new CustomJ48();
		Instances data = CustomJ48.loadDatasetArff(dataset);
		
		data.setClass(data.attribute(data.numAttributes() - 1));

		j48.buildClassifier(data);
		System.out.println(j48);
		Instances instances = j48.splitNumericSuppliedTest(data);
		System.out.println(instances);
		System.out.println(data.classAttribute().value((int) j48.classifyInstance(instances.firstInstance())));
		
		
		J48 tree = new J48();
		tree.buildClassifier(data);
		System.out.println(tree);
		
		
	}
}

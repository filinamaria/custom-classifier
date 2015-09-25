package weka.custom_classifier.Id3;

import java.io.File;
import java.io.IOException;
import java.util.Enumeration;

import com.google.common.math.DoubleMath;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.converters.ArffLoader;
import weka.core.Instance;
import weka.core.Instances;
import weka.custom_classifier.Tree;

public class CustomId3 extends Classifier
{
	/** For serialization */
	private static final long serialVersionUID = 1L;
	private Tree decisionTree;
	
	/**
	 * Default Constructor
	 */
	public CustomId3()
	{
		decisionTree = new Tree();
	}
	
	/**
	 * @return capabilities
	 */
	private Capabilities classifierCapabilities()
	{
		Capabilities capabilities = super.getCapabilities();
		
		// set capabilities
		capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
		capabilities.enable(Capability.NOMINAL_CLASS);
		capabilities.enable(Capability.MISSING_CLASS_VALUES);
		capabilities.setMinimumNumberInstances(0);
		
		return capabilities;
	}
	
	/**
	 * Build ID3 Classifier
	 * @param data training data
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception 
	{
		classifierCapabilities().testWithFail(data);
		
		data.deleteWithMissingClass();
                ArrayList<Attribute> selectedAttr = new ArrayList();
		generateTree(data, decisionTree, selectedAttr);
	}
	
	/**
	 * Classify instance
	 */
	public double classifyInstance(Instance instance)
	{
		return classifyInstance(instance, decisionTree);
	}
	
	/**
	 * Classify instance from certain node of decisionTree
	 * @param instance
	 * @param tree
	 * @return
	 */
	private double classifyInstance(Instance instance, Tree tree)
	{
		if(tree.getAttribute() == null){
			return tree.getClassValue();
		}else{
			return classifyInstance(instance, tree.getChild((int) instance.value(tree.getAttribute())));
		}
	}
	
	/**
	 * Generate ID3 (recursive tree)
	 * @param data training data
	 * @param tree node
         * @param selectedAttr already selected attribute at parents node
	 */
	public void generateTree(Instances data, Tree tree, ArrayList<Attribute> selectedAttr)
	{
		//handle instances with missing values
		if(data.numInstances() == 0)
		{
			tree.setAttribute(null);
			tree.setClassValue(Instance.missingValue());
			return;
		}
		
		Enumeration attributes = data.enumerateAttributes();
		double[] infoGains = new double[data.numAttributes()];
	
		//information gain calculation
		while(attributes.hasMoreElements()){
			Attribute attribute = (Attribute) attributes.nextElement();
                        if (!selectedAttr.contains(attribute)) {
                            infoGains[attribute.index()] = informationGain(data, attribute);
                            System.out.println(infoGains[attribute.index()]);
                        }
                        else infoGains[attribute.index()]=0.0;
		}
		
		Attribute highestIGAtt = data.attribute(maxIndex(infoGains));
		selectedAttr.add(highestIGAtt);
		//build decision tree
		tree.setAttribute(highestIGAtt);
		
		//leaf detection
		if(Double.compare(infoGains[highestIGAtt.index()], 0.0) == 0) //leaf
		{
			tree.setAttribute(null);
			double[] distribution = new double[data.numClasses()];
			Enumeration instances = data.enumerateInstances();
			
			while(instances.hasMoreElements())
			{
				Instance instance = (Instance) instances.nextElement();
				distribution[(int) instance.classValue()]++;
			}
			
			tree.setClassValue(maxIndex(distribution));
			tree.setClassAttribute(data.classAttribute());
		}
		else //not leaf yet, generate child
		{ 
			Instances[] splittedData = split(data, highestIGAtt);
			Tree[] children = new Tree[tree.getAttribute().numValues()];
			
			for(int i = 0; i < children.length; i++)
			{
				children[i] = new Tree();
				tree.addChildren(children);
				generateTree(splittedData[i], children[i], selectedAttr);
			}
		}
	}
	
	/**
	 * Split instances 
	 * @param data
	 * @param att
	 * @return
	 */
	private Instances[] split(Instances data, Attribute att)
	{
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
	 * @return array's index which hold highest value 
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
	 * Calculate information gain
	 * @param data training data
	 * @param att attribute
	 * @return information gain
	 */
	private double informationGain(Instances data, Attribute att){
		double informationGain = entropy(data);
		int numOfLabels = att.numValues();

		Instances[] instancesDistribution = new Instances[numOfLabels];
		
		for(int i = 0; i < instancesDistribution.length; i++)
		{
			instancesDistribution[i] = new Instances(data, data.numInstances());
			instancesDistribution[i].delete();
		}
		
		for(int i = 0; i < data.numInstances(); i++)
		{			
			instancesDistribution[(int) data.instance(i).value(att)].add(data.instance(i));
		}
		
		for(int i = 0; i < numOfLabels; i++)
		{
			double numInstancesOfLabel = (double) instancesDistribution[i].numInstances();
			informationGain -=  numInstancesOfLabel / (double) data.numInstances() * entropy(instancesDistribution[i]);
		}
		
		return informationGain;
	}
	
	/**
	 * Entropy calculation
	 * @param data training data
	 * @return entropy of data
	 */
	private double entropy(Instances data)
	{
		double entropy = 0.0;
		int numOfClasses = data.classAttribute().numValues();
		int[] numOfInstancesPerClass = new int[numOfClasses];
		
		for(int i = 0; i < data.numInstances(); i++)
		{
			numOfInstancesPerClass[(int) data.instance(i).classValue()]++;
		}
		
		for(int i = 0; i < numOfClasses; i++)
		{
			if(numOfInstancesPerClass[i] != 0)
			{
				double temp = (double) numOfInstancesPerClass[i] / (double) data.numInstances();
				entropy -= temp * DoubleMath.log2(temp);
			}
		}
		
		return entropy;
	}
	
	/**
	 * Convert ID3 to string
	 * @param level
	 * @param tree
	 * @return string
	 */
	private String toString(int level, Tree tree) {
	    StringBuffer text = new StringBuffer();
	    
	    if (tree.getAttribute() == null) 
	    {
	      if (Instance.isMissingValue(tree.getClassValue()))
	      {
	        text.append(": null");
	      } 
	      else 
	      {
	        text.append(": " + tree.getClassAttribute().value((int) tree.getClassValue()));
	      } 
	    } 
	    else 
	    {
	      for (int j = 0; j < tree.getAttribute().numValues(); j++) 
	      {
	        text.append("\n");
	        for (int i = 0; i < level; i++) 
	        {
	          text.append("|  ");
	        }
	        text.append(tree.getAttribute().name() + " = " + tree.getAttribute().value(j));
	        text.append(toString(level + 1, tree.getChild(j)));
	      }
	    }
	    return text.toString();
	}
	
	/**
	 * @return ID3 to string
	 */
	public String toString() {
	    if ((decisionTree.getAttribute() == null) && (decisionTree.getChildren() == null)) 
	    {
	      return "Id3: No model built yet.";
	    }
	    return "Id3\n\n" + toString(0, decisionTree);
	}
	
	/**
	 * For testing purpose
	 * @param filePath
	 * @return
	 * @throws IOException
	 */
	public static Instances loadDatasetArff(String filePath) throws IOException
    { 
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filePath));
		return loader.getDataSet();
    }
	
	/**
	 * For testing purpose only
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception{
		String dataset = "example/weather.nominal.arff";
		CustomId3 id3 = new CustomId3();
		Instances data = CustomId3.loadDatasetArff(dataset);
		
		data.setClass(data.attribute(data.numAttributes() - 1));

		id3.buildClassifier(data);
		
		Instance instance = data.firstInstance();
		//System.out.println(instance);
		//System.out.println(data.classAttribute().value((int) id3.classifyInstance(instance)));
		System.out.println("Custom made ID3");
		System.out.println(id3);
		Id3 tree = new Id3();
		tree.buildClassifier(data);
		System.out.println("Weka's ID3");
		System.out.println(tree);	
	}
}

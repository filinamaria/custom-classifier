/*
 * Jan Wira Gotama Putra / 13512015
 * Mario Filino / 13512055
 * Melvin Fonda / 13512085
 */
package weka.custom_classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author 3Cisitus
 */
public class ExplorationClassifier {
    
    private Instances dataTrain;
    private Instances dataTrainStructure;
    private NaiveBayesUpdateable naiveBayes;
    private J48 decisionTree;
    private int lastBuiltClassifier; //1 for Naive Bayes, 2 for J48, 0 for not yet build
    private int classAttributeIndex;
    
    /**
     * Default Constructor
     */
    public ExplorationClassifier()
    {
        lastBuiltClassifier = 0;
        decisionTree = null;
        naiveBayes = null;
        dataTrain = null;
        classAttributeIndex = 0;
    }
    
    /**
     * 
     */
    public void printDataTrain()
    {
        System.out.println(dataTrain.toString());
        System.out.println(dataTrainStructure.toString());
    }
    
    /**
     * 
     */
    public void printHypothesis()
    {
        if (this.lastBuiltClassifier==0)
            System.out.println("No hypothesis yet");
        else if (this.lastBuiltClassifier==1)
            System.out.println(naiveBayes.toString());
        else System.out.println(decisionTree.toString());
    }
    
    /**
     * 
     * @param classAttributeIndex the classAttributeIndex for dataTrain
     */
    public void setDataTrainClassAttributeIndex(int classAttributeIndex)
    {
        this.classAttributeIndex = classAttributeIndex;
        dataTrain.setClassIndex(classAttributeIndex);
        dataTrainStructure.setClassIndex(classAttributeIndex);
    }
    
    /**
    * @param filePath the file path of dataset
    * @param mode the mode=1 for arff, mode=2 for csv
    */
    public void loadDataset(String filePath, int mode)
    { 
        if (mode==1) //load arff
        {
            try {
                ArffLoader loader = new ArffLoader();
                loader.setSource(new File(filePath));
                dataTrain = loader.getDataSet();
                dataTrainStructure = loader.getStructure();
            } catch (IOException ex) {
                System.out.println("File not Found!");
            }
        }
        else if (mode==2) //load csv
        {
            try {
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(filePath));
                dataTrain = loader.getDataSet();
                dataTrainStructure = loader.getStructure();
            } catch (IOException ex) {
                System.out.println("File not Found!");
            }
        }
        else System.out.println("Mode not Supported!");
    }
    
    /**
     * Delete attribute from index Range selection
     * @param attrIndexRange, the attribute index range selection which want to be removed
     */
    public void removeAttributes(String attrIndexRange)
    {
        try {
            Remove remove = new Remove();
            remove.setAttributeIndices(attrIndexRange);
            remove.setInvertSelection(false);
            remove.setInputFormat(dataTrain);
            dataTrain = Filter.useFilter(dataTrain, remove);
            dataTrainStructure = Filter.useFilter(dataTrainStructure, remove);
        } catch (Exception ex) {
            System.out.println("Cannot remove attribute!");
        }
    }
    
    /**
     * 
     * @param resampleOptions the resampleOptions. The valid options could be seen from http://weka.sourceforge.net/doc.dev/weka/filters/supervised/instance/Resample.html
     */
    public void resample(String resampleOptions)
    {
       
        try {
            Resample sampler = new Resample();
            sampler.setOptions(weka.core.Utils.splitOptions(resampleOptions));
            sampler.setInputFormat(dataTrain);
            dataTrain = Filter.useFilter(dataTrain, sampler);
        } catch (Exception ex) {
            Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * @param evalMode the evalMode=1 for 10-fold cross, evalMode=2 for percentageSplit
     * @param percentageSplit the percentageSplit for training and testing, percentageSplit*dataTrain used for training, the remaining used for testing
     */
    public void buildClassifierNaiveBayes(int evalMode, double percentageSplit)
    {
        try {
            this.lastBuiltClassifier = 1;  
            
            // split data between data for training and testing
            Instances[] trainTest = this.splitData(evalMode, percentageSplit, dataTrain);
            
            // train NaiveBayes
            naiveBayes = new NaiveBayesUpdateable();
            naiveBayes.buildClassifier(dataTrainStructure);     
            int N = trainTest[0].numInstances();
            for (int i=0; i<N; i++) {
                naiveBayes.updateClassifier(trainTest[0].instance(i));
            }
            
            //training evaluation
            if (trainTest[1].numInstances()>0)
                trainingEvaluation(naiveBayes, evalMode, trainTest[1], percentageSplit);         
        } catch (Exception ex) {
            Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * 
     * @param options the options for building J48 based classifier
     * @param evalMode the evalMode=1 for 10-fold cross, evalMode=2 for percentageSplit
     * @param percentageSplit the percentageSplit for training and testing, percentageSplit*dataTrain used for training, the remaining used for testing
     */
    public void builClassifierDecisionTree(String options, int evalMode, double percentageSplit)
    {
        try {
            this.lastBuiltClassifier = 2;
            // split data for training and testing
            Instances[] trainTest = this.splitData(evalMode, percentageSplit, dataTrain);
            
            // train decision tree
            decisionTree = new J48(); // new instance of tree
            decisionTree.setOptions(Utils.splitOptions(options)); // set the options
            decisionTree.buildClassifier(trainTest[0]); // build classifier
            
            // evaluate decision tree
            if (trainTest[1].numInstances()>0)
                trainingEvaluation(decisionTree, evalMode, trainTest[1], percentageSplit);
        } catch (Exception ex) {
            Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * Split data for training and testing
     * @param evalMode the evalMode=1 for 10-fold cross, evalMode=2 for percentageSplit
     * @param percentageSplit the percentageSplit for training and testing, percentageSplit*dataTrain used for training, the remaining used for testing
     * @return Instances[], Instances[0] for training, Instances[1] for testing
     */
    private Instances[] splitData(int evalMode, double percentageSplit, Instances dataTrain)
    {
        Instances train=null, test=null;
        if (evalMode==2) //eval PercentageSplit
        {
            int trainSize = (int) Math.round(dataTrain.numInstances() * percentageSplit);
            int testSize = dataTrain.numInstances() - trainSize;
            train = new Instances(dataTrain, 0, trainSize);
            test = new Instances(dataTrain, trainSize, testSize);
        }
        else { //10 fold cross
            train = dataTrain;
            test = dataTrain;
        }
        Instances[] retVal = {train, test};
        return retVal;
    }
    /**
     * 
     * @param cls, the classifier
     * @param evalMode the evalMode=1 for 10-fold cross, evalMode=2 for percentageSplit
     * @param test, the test set
     * @param percentageSplit the percentageSplit for training and testing, percentageSplit*dataTrain used for training, the remaining used for testing
     */
    private void trainingEvaluation(Classifier cls, int evalMode, Instances test, double percentageSplit)
    {
        try {
            Evaluation eval = new Evaluation(test);
            if (evalMode==1) { //10 fold cross
                eval.crossValidateModel(cls, test, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nEvaluation Results 10-fold-cross", false));
            }
            else if (evalMode==2) { //percentage split
                eval.evaluateModel(cls, test);
                System.out.println(eval.toSummaryString("\nEvaluation Results PercentageSplit "+Double.toString(percentageSplit), false));
            }
        } catch (Exception ex) {
            Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * Test model using supplied test set
     * @param dataTestPath dataTest file path
     * @param mode the mode=1 for arff, 2 for CSV
     * @param classAttributeIndex for the dataTest attributeIndex
     */
    public void testModelSuppliedTest(String dataTestPath, int mode, int classAttributeIndex)
    {
        try {
            Instances dataTest = null;
            if (mode==1) { //arff file
                ArffLoader loader = new ArffLoader();
                loader.setSource(new File(dataTestPath));
                dataTest = loader.getDataSet();
            }
            else if (mode==2) { //csv file
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(dataTestPath));
                dataTest = loader.getDataSet();
            }
            dataTest.setClassIndex(classAttributeIndex);
            
            Evaluation eval = new Evaluation(dataTest);
            Classifier cls = null;
            if (this.lastBuiltClassifier==1)
                cls = naiveBayes;
            else if (this.lastBuiltClassifier==2)
                cls = decisionTree;
            
            // evaluation using supplied test set (all)
            eval.evaluateModel(cls, dataTest);
            System.out.println(eval.toSummaryString("\nTesting Results Using Supplied Test Set", false));
        } catch (Exception ex) {
            Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * Save Classifier Model
     */
    public void saveModel()
    {
        if (lastBuiltClassifier==0) {
            System.out.println("No classifier built yet");
        }
        else if (lastBuiltClassifier==1) {
            try {
                saveHypothesis(naiveBayes);
            } catch (Exception ex) {
                Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else if (lastBuiltClassifier==2) {
            try {
                saveHypothesis(decisionTree);
            } catch (Exception ex) {
                Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
     * 
     * @param clsNum the classifier, 1 for naiveBayes, 2 for decisionTree 
     */
    public void loadModel(int clsNum)
    {
        if (clsNum==1) //naive Bayes
        {
            try {
                this.lastBuiltClassifier = 1;
                naiveBayes = new NaiveBayesUpdateable();
                naiveBayes = (NaiveBayesUpdateable) weka.core.SerializationHelper.read("class weka.classifiers.bayes.NaiveBayes.model");
            } catch (Exception ex) {
                Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else if (clsNum==2) { //decision tree
            try {
                this.lastBuiltClassifier = 2;
                decisionTree = new J48();
                decisionTree = (J48) weka.core.SerializationHelper.read("class weka.classifiers.trees.J48.model");
            } catch (Exception ex) {
                Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
     * Read class attribute number saved in external file
     * @throws Exception 
     */
    private void readClassAttrNum() throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("classAttribute.txt"));
        String line = null;
        while ((line = reader.readLine()) != null) {
            this.classAttributeIndex = Integer.parseInt(line);
        }
    }
    
    /**
     * Classify unlabeled data from external file
     * @param filePath the unlabeled data file
     * @param mode the mode=1 for arff, 2 for CSV
     * @param classAttrIdx class attribute index for unlabeled data
     */
    public void classifyUnlabeledData(String filePath, int mode, int classAttrIdx)
    {
        Instances unlabeled = null, labeled = null;
        if (mode!=1 && mode!=2)
        {
            System.out.println("Mode unsupported!");
        }
        else {
            if (mode==1) //load arff
            {
                try {
                    ArffLoader loader = new ArffLoader();
                    loader.setSource(new File(filePath));
                    unlabeled = loader.getDataSet();
                } catch (IOException ex) {
                    System.out.println("File not Found!");
                }
            }
            else //load csv
            {
                try {
                    CSVLoader loader = new CSVLoader();
                    loader.setSource(new File(filePath));
                    unlabeled = loader.getDataSet();
                } catch (IOException ex) {
                    System.out.println("File not Found!");
                }
            }
            try {
                unlabeled.setClassIndex(classAttrIdx);
                labeled = new Instances(unlabeled);
                
                double clsLabel = 0;
                Classifier cls = null;
                if (this.lastBuiltClassifier==1)
                    cls = naiveBayes;
                else if (this.lastBuiltClassifier==2)
                    cls = decisionTree;
                
                //give label to unlabeled data
                for (int i = 0; i < unlabeled.numInstances(); i++) {
                    clsLabel = cls.classifyInstance(unlabeled.instance(i));
                    labeled.instance(i).setClassValue(clsLabel);
                }
                
                //output the result
                System.out.println("Classified Instances Label");
                ConverterUtils.DataSink.write(System.out, labeled);
                System.out.println();
            } catch (Exception ex) {
                Logger.getLogger(ExplorationClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
     * 
     * @param cls the classifier
     * @throws Exception 
     */
    private void saveHypothesis(Classifier cls) throws Exception {
    /* I.S : cls defined
       F.S : cls saved to external file */
        
        weka.core.SerializationHelper.write(cls.getClass().toString()+".model", cls);
        
        //save class attribute
        FileWriter fw = new FileWriter("classAttribute.txt");
        PrintWriter pw = new PrintWriter(fw);

        //Write to file line by line
        pw.println(classAttributeIndex);
        //Flush the output to the file
        pw.flush();
        //Close the Print Writer
        pw.close();
        //Close the File Writer
        fw.close(); 
    }
}

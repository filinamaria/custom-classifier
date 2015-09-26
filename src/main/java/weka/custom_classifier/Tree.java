package weka.custom_classifier;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;

public class Tree {
	private Tree[] children;
	private Attribute attribute;
	private double classValue;
	private Attribute classAttribute;
	private double[] probs; //probability distribution of each attribute value -> for J48
        
	public Tree(){}
	
	public void setAttribute(Attribute att){
		attribute = att;
	}
	
	public void setClassValue(double value){
		classValue = value;
	}
	
	public void setClassAttribute(Attribute att){
		classAttribute = att;
	}
		
	public Attribute getAttribute(){
		return attribute;
	}
	
	public double getClassValue(){
		return classValue;
	}
	
	public Attribute getClassAttribute(){
		return classAttribute;
	}
	
	public Tree[] getChildren(){
		return children;
	}
	
	public Tree getChild(int index){
		return children[index];
	}
	
	public void addChild(int index, Tree child){
		children[index] = child;
	}
	
	public void addChildren(Tree[] children){
		this.children = children;
	}
        
        public void addProbs(double[] prop) {
            this.probs = prop;
        }
        
        public void addProb(int index, double prop) {
            this.probs[index] = prop;
        }
        
        public double getProb(int index) {
            return probs[index];
        }
        
        public double[] getProbs() {
            return probs;
        }
}

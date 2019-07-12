package de.tudarmstadt.ukp.experiments.argumentation.convincingness.svmlib;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.SortedMap;
import java.util.SortedSet;

import org.apache.uima.UIMAException;

public class FeatureNameWriter {

	public static void main(String[] args) throws UIMAException, ClassNotFoundException, IOException {
        File inputDir = new File("/home/local/UKP/simpson/data/personalised_argumentation/tempdata/all2");
        SortedSet<String> sentimentFeatureNames = SVMLibExporter.extractFeatureNames(inputDir);
		        
        inputDir = new File("/home/local/UKP/simpson/data/personalised_argumentation/tempdata/all3");
        SortedSet<String> featureNames = SVMLibExporter.extractFeatureNames(inputDir);
        SortedMap<String, Integer> mapping = SVMLibExporter.mapFeaturesToInts(featureNames);        

        PrintWriter out = new PrintWriter("/home/local/UKP/simpson/data/personalised_argumentation/tempdata/feature_names_all2.txt");
        out.println(sentimentFeatureNames);
        for(String featName : sentimentFeatureNames){ 
        	out.print(mapping.get(featName));
        	out.print(", ");
        }
        out.close();
        
        out = new PrintWriter("/home/local/UKP/simpson/data/personalised_argumentation/tempdata/feature_names_all3.txt");
        out.println(featureNames);
        for(String featName2 : featureNames){ 
        	out.print(mapping.get(featName2));
        	out.print(", ");   
        }
        out.close();
	}

}

/*
 * Copyright 2016
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.tudarmstadt.ukp.experiments.argumentation.convincingness.svmlib;

import de.tudarmstadt.ukp.dkpro.core.io.bincas.BinaryCasReader;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;

import java.io.*;
import java.util.*;

/**
 * @author Ivan Habernal
 */
public class SVMLibEverythingExporter extends SVMLibExporter
{
    public static void main(String[] args)
            throws Exception
    {
        File inputDir = new File(args[0]);
        File outputDir = new File(args[1]);
        
        File featureDir = inputDir;
        
        if (args.length > 2) {
        	featureDir = new File(args[2]);
        }

        SortedSet<String> featureNames = extractFeatureNames(featureDir);

        //System.out.println(featureNames);
        
        SortedMap<String, Integer> mapping = mapFeaturesToInts(featureNames);

        String mappingFile = "/tmp/mapping.bin";
        ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(mappingFile));
        os.writeObject(mapping);
        os.close();
        
        String[] fileNames = inputDir.list();

        int counter = 0;
        for (String fileName : fileNames) {
            counter++;

            String[] fileNameToks = fileName.split("\\.");
            if (!fileNameToks[fileNameToks.length-2].equals("bin") || !fileNameToks[fileNameToks.length-1].equals("bz2")){
                System.out.println("Skipping a file I don't think is relevant: " + fileName);            	
            	continue;
            }
            
            String foldName = fileNameToks[0];
            
            System.out.println("Processing " + foldName);

            // generate training data
            SimplePipeline.runPipeline(
                    CollectionReaderFactory.createReaderDescription(

                            BinaryCasReader.class,
                            BinaryCasReader.PARAM_SOURCE_LOCATION,
                            inputDir,
                            BinaryCasReader.PARAM_PATTERNS,
                            BinaryCasReader.INCLUDE_PREFIX + fileName //+ "*.bz2"
                    ),
                    AnalysisEngineFactory.createEngineDescription(
                            LIBSVMFileProducer.class,
                            LIBSVMFileProducer.PARAM_FEATURES_TO_INT_MAPPING,
                            mappingFile,
                            LIBSVMFileProducer.PARAM_OUTPUT_FILE,
                            new File(outputDir, foldName + ".libsvm.txt")
                    )

            );
        }
    }
}

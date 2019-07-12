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

package de.tudarmstadt.ukp.experiments.argumentation.convincingness.sampling;

import de.tudarmstadt.ukp.experiments.argumentation.convincingness.sampling.AnnotatedArgumentPair.MTurkAssignment;
import java.io.File;
import java.util.*;

/**
 * This is adapted from Step5GoldLabelEstimator. Instead of using MACE to produce a gold label, we take the answer 
 * given by a single crowd worker. This produces a noisy dataset containing real human errors. 
 * @author Ivan Habernal/Edwin Simpson
 */
public class RandomCrowdSample
{
    @SuppressWarnings("unchecked")
    public static void main(String[] args)
            throws Exception
    {
        String inputDir = args[0];
        File outputDir = new File(args[1]);

        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        // we will process only a subset first
        List<AnnotatedArgumentPair> allArgumentPairs = new ArrayList<>();

        Collection<File> files = IOHelper.listXmlFiles(new File(inputDir));

        for (File file : files) {
            allArgumentPairs
                    .addAll((List<AnnotatedArgumentPair>) XStreamTools.getXStream().fromXML(file));
        }
        
        for (AnnotatedArgumentPair pair : allArgumentPairs) {
        	List<MTurkAssignment> assmts = pair.mTurkAssignments;
        	pair.mTurkAssignments = new ArrayList<MTurkAssignment>();
        	pair.mTurkAssignments.add(assmts.get(0)); // we keep only the first assignment for each pair
        }
        
        // assign the gold label
        for (int i = 0; i < allArgumentPairs.size(); i++) {
            AnnotatedArgumentPair annotatedArgumentPair = allArgumentPairs.get(i);
            String goldLabel = annotatedArgumentPair.mTurkAssignments.get(0).getValue();
            annotatedArgumentPair.setGoldLabel(goldLabel);
        }

        // now sort the data back according to their original file name
        Map<String, List<AnnotatedArgumentPair>> fileNameAnnotatedPairsMap = new HashMap<>();
        for (AnnotatedArgumentPair argumentPair : allArgumentPairs) {
            String fileName = IOHelper.createFileName(argumentPair.getDebateMetaData(),
                    argumentPair.getArg1().getStance());

            if (!fileNameAnnotatedPairsMap.containsKey(fileName)) {
                fileNameAnnotatedPairsMap.put(fileName, new ArrayList<AnnotatedArgumentPair>());
            }

            fileNameAnnotatedPairsMap.get(fileName).add(argumentPair);
        }

        // and save them to the output file
        for (Map.Entry<String, List<AnnotatedArgumentPair>> entry : fileNameAnnotatedPairsMap
                .entrySet()) {
            String fileName = entry.getKey();
            List<AnnotatedArgumentPair> argumentPairs = entry.getValue();

            File outputFile = new File(outputDir, fileName);

            // and save all sampled pairs into a XML file
            XStreamTools.toXML(argumentPairs, outputFile);

            System.out.println("Saved " + argumentPairs.size() + " pairs to " + outputFile);
        }

    }
}

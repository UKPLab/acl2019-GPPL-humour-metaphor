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

package de.tudarmstadt.ukp.experiments.argumentation.convincingness.preprocessing;

import de.tudarmstadt.ukp.experiments.argumentation.convincingness.io.ArgumentPairToSingleReader;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;

import java.io.IOException;

/**
 * Extends class by Ivan Habernal to produce separate outputs for each unique argument in a dataset containing 
 * argument pairs.
 * @author Edwin Simpson
 */
public class PipelineSeparateArguments extends Pipeline
{
    public static void processDataPairs(String inputDir, String outputDir)
            throws IOException, UIMAException
    {
        SimplePipeline.runPipeline(
                CollectionReaderFactory.createReaderDescription(
                        ArgumentPairToSingleReader.class,
                        ArgumentPairToSingleReader.PARAM_SOURCE_LOCATION,
                        inputDir,
                        ArgumentPairToSingleReader.PARAM_PATTERNS,
                        ArgumentPairToSingleReader.INCLUDE_PREFIX + "*.csv"
                ),
                createPipeline(outputDir)
        );
    }

    public static void main(String[] args)
            throws IOException, UIMAException
    {
        processDataPairs(args[0], args[1]);
        // commented so we can sort out the ranking data separately
        //processDataRank(args[2], args[3]);
    }
}

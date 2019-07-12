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

package de.tudarmstadt.ukp.experiments.argumentation.convincingness.io;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.jcas.JCas;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Extended from class by Ivan Habernal to produce separate output for each unique argument encountered,
 * determined by the argument ID. 
 * @author Edwin Simpson
 */
public class ArgumentPairToSingleReader
        extends ArgumentPairReader
{
	private ArrayList<String> idsAlreadySeen = new ArrayList<String>();  
	
    @Override public boolean hasNext()
            throws IOException, CollectionException
    {
        boolean hasMoreText = super.hasNext();
        if (!hasMoreText){
        	return false;
        }
        if (currentLines.isEmpty()) {
            loadNextFile();
        }
        
	    while(!currentLines.isEmpty()) {
	        // see if there are any lines with unseen arguments
	        String line = currentLines.peek();
	        
	        //System.out.println(line);
	        
	        String[] split = line.split("\t");
	        String id = split[0].trim();
	        
	        String[] idsplit = id.split("_");
	        String id1 = idsplit[0];
	        String id2 = "";
	        
	        if (idsplit.length > 1) {
	        	id2 = idsplit[1];
	        }
	        
	        if (!idsAlreadySeen.contains(id1)) {
	        	return true;
	        }
	        else if (!id2.equals("") && !idsAlreadySeen.contains(id2)){
	        	return true;
	        } else {
	        	// we can remove this line from the list with no problem
	        	currentLines.remove();
	        }
	    }
	    return super.hasNext(); // no ids were found that were not already seen
    }	
	
    @Override
    public void getNext(CAS aCAS)
            throws IOException, CollectionException
    {
        try {
            if (currentLines.isEmpty()) {
                loadNextFile();
            }

            String line = currentLines.poll();
            String[] split = line.split("\t");
            String id = split[0].trim();
            
            String[] idsplit = id.split("_");
            
            String id1 = idsplit[0];
            String id2 = "";
            
            if (idsplit.length > 1) {   		
            	id2 = idsplit[1];
            }
            
            if (idsAlreadySeen.contains(id1) && (id2.equals("") || idsAlreadySeen.contains(id2)) ){
            	if (hasNext()){
            		getNext(aCAS);
            	}
            	return;
            }
            
            int skip = 0;
            if (split.length > 4){
            	skip = 1; // skip over column 1 as there is an extra field here
            }
            //String label = split[1+skip].trim(); // why do we need this to identify the argments?
            String a1 = split[2+skip].trim();
            String a2 = split[3+skip].trim();

            // set label
            JCas jCas = aCAS.getJCas();
            DocumentMetaData metaData = DocumentMetaData.create(jCas);
            
            String name;
            if (!idsAlreadySeen.contains(id1)){
            	name = id1; //currentFileName + "_" + id + "_" + label;
                // set content
                jCas.setDocumentText(a1);// + "\n" + a2);
                jCas.setDocumentLanguage("en"); 
           
                //record which arguments we have seen
                idsAlreadySeen.add(id1);
                
                if (!idsAlreadySeen.contains(id2)){
                	//must also add second argument -- do this by setting putting the current line back into the queue,
                	// next time it will skip argument 1 as it is already added.
                	currentLines.add(line);
                }
            } else {
            	name = id2; //currentFileName + "_" + id + "_" + label;
            	//record that we have seen this argument
            	idsAlreadySeen.add(id2);
                // set content
                jCas.setDocumentText(a2);// + "\n" + a2);
                jCas.setDocumentLanguage("en");
            }
            metaData.setDocumentTitle(name);
            metaData.setDocumentId(name);
        }
        catch (CASException e) {
            throw new CollectionException(e);
        }
    }
}
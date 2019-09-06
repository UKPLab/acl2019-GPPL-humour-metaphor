# Predicting Humorousness and Metaphor Novelty with Gaussian Process Preference Learning

This project contains source code and data for a Gaussian Process
Preference Learning (GPPL) system for predicting humorousness and
metaphor novelty.

If you reuse the software or data, please use the following citation:

> Edwin Simpson, Erik-Lân Do Dinh, Tristan Miller, and Iryna
> Gurevych. [Predicting humorousness and metaphor novelty with
> Gaussian process preference
> learning](https://fileserver.ukp.informatik.tu-darmstadt.de/UKP_Webpage/publications/2019/2019_ACL_GPPL_for_Funniness_and_Metaphor_Novelty_Ranking.pdf). In
> [Proceedings of the 57th Annual Meeting of the Association for
> Computational Linguistics (ACL 2019)](http://www.acl2019.org/),
> July 2019, pp. 5716–5728.

> **Abstract:** The inability to quantify key aspects of creative
> language is a frequent obstacle to natural language understanding.
> To address this, we introduce novel tasks for evaluating the
> creativeness of language—namely, scoring and ranking text by
> humorousness and metaphor novelty.  To sidestep the difficulty of
> assigning discrete labels or numeric scores, we learn from pairwise
> comparisons between texts.  We introduce a Bayesian approach for
> predicting humorousness and metaphor novelty using Gaussian process
> preference learning (GPPL), which achieves a Spearman's ρ of 0.56
> against gold using word embeddings and linguistic features.  Our
> experiments show that given sparse, crowdsourced annotation data,
> ranking using GPPL outperforms best–worst scaling. We release a new
> dataset for evaluating humour containing 28,210 pairwise comparisons
> of 4030 texts, and make our software freely available.

```
@inproceedings{simpson2019predicting,
   author    = {Edwin Simpson and Do Dinh, Erik-L{\^{a}}n and
                Tristan Miller and Iryna Gurevych},
   title     = {Predicting Humorousness and Metaphor Novelty with
                {Gaussian} Process Preference Learning},
   booktitle = {Proceedings of the 57th Annual Meeting of the
                Association for Computational Linguistics (ACL 2019)},
   month     = jul,
   year      = {2019},
   pages     = {5716--5728},
}
```

## Dependencies

* Python 3

For running the experiments, please see the requirements.txt. 
We recommend using a virtual environment to install the packages:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt 
```

## Project Structure

* data -- a folder containing small data files + default place to generate dataset files for the experiments
* python -- the scripts for running the experiments
* python/humour_dataset -- script used to remove workers who failed a captcha test, i.e. to detect spammers
* python/models -- the implementation of the GPPL method
* python/preprocessing -- methods used to extract frequency and bigram features
* python/tools -- 
* results -- an output directory for storing results

## How to run the experiments

Before running the experiments which use word embeddings, please unzip the file `data/embeddings.tar.bz2` .
Before running the experiments with the metaphor novelty dataset, please unzip the file `data/all.zip`.

### Task 1

For the humour dataset, run GPPL:
```
python run_task1_experiment.py humour
```
Now, look in the results directory to see the output from the first step. 
In `Task_1_analysis_humour.py`, change the resfile variable to point to the new results csv file.
Next, run:
```
python Task_1_analysis_humour.py
```
This will produce plots in results directory and output a number of correlation measures shown in the paper.
It will also show classification performance in separating funny and non-funny texts (using the AUC metric),
and inter-annotator agreement.

The same process can be used for metaphor novelty data:
```
python run_task1_experiment.py metaphor
```
Now, set the resfile variable in `Task_1_analysis_metaphor.py` to the results csv file from the previous step, which 
is in the results directory. Then run:
```
python Task_1_analysis_metaphor.py
```
This will produce another plot in the results directory and output correlation measures to the command line.

### Task 2

For the humour dataset:
```
python run_experiments.py humour
```

For the metaphor dataset:
```
python run_experiments.py metaphor
```

### Task 3

For the humour dataset:
```
python run_experiments.py humour 0.05,0.1,0.2,0.33,0.66 
```

For the metaphor dataset:
```
python run_experiments.py metaphor 0.05,0.1,0.2,0.33,0.66
```

GPPL will be tested with both
'annotation' and 'pairs' strategies, as described in the paper.
The model will be trained with frequency, ngram and average word embeddings features.

### Task 4

For the humour dataset:
```
python run_experiments.py humour 0.05,0.1,0.2,0.33,0.66 task4 
```

For the metaphor dataset:
```
python run_experiments.py metaphor 0.05,0.1,0.2,0.33,0.66 task4 
```

## Copyright and licensing

Copyright © 2019 Ubiquitous Knowledge Processing Lab, Technische
Universität Darmstadt.

The software contributed by the UKP Lab is licensed under the [Apache
License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The annotations and this documentation are licensed under a [Creative
Commons Attribution 4.0 International
License](https://creativecommons.org/licenses/by/4.0/) (CC-BY).

To the best of our knowledge, the individual texts in the
[SemEval-2017 humour
data](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/sense_labelling_resources/sense_annotated_english_puns/index.en.jsp)
set we redistribute here are not eligible for copyright.  Please refer
to the licence information in [the original data set
distribution](http://alt.qcri.org/semeval2017/task7/data/uploads/semeval2017_task7.tar.xz)
for further details.

## Contact

**Contact person:** Edwin Simpson,
simpson@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something
is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for
> the sole purpose of giving additional background details on the
> respective publication.

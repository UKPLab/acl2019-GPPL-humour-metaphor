## Dependencies

Dependencies for running gp_pref_learning model:

   * scikit-learn==0.18.1
   * scipy==0.19.0
   * numpy==1.12.1

For running the experiments, please see the requirements.txt for further dependencies. 

## How to run

We introduce a scalable Bayesian preference
learning method for identifying convincing ar-
guments in the absence of gold-standard rat-
ings or rankings. In contrast to previous work,
we avoid the need for separate methods to
perform quality control on training data, pre-
dict rankings and perform pairwise classifica-
tion. Bayesian approaches are an effective so-
lution when faced with sparse or noisy train-
ing data, but have not previously been used
to identify convincing arguments. One issue
is scalability, which we address by develop-
ing a stochastic variational inference method
for Gaussian process (GP) preference learn-
ing. We show how our method can be ap-
plied to predict argument convincingness from
crowdsourced data, outperforming the previ-
ous state-of-the-art, particularly when trained
with small amounts of unreliable data. We
demonstrate how the Bayesian approach en-
ables more effective active learning, thereby
reducing the amount of data required to iden-
tify convincing arguments for new users and
domains. While word embeddings are princi-
pally used with neural networks, our results
show that word embeddings in combination
with linguistic features also benefit GPs when
predicting argument convincingness.

**Contact person:** Edwin Simpson, simpson@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be)
or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background
details on the respective publication.

## Project Structure

* data -- a folder containing small data files + default place to generate dataset files for the experiments
* documents -- sources for the paper
* error_analysis -- working data files for error analysis
* python/analysis -- experiment code
* python/analysis/habernal_comparison -- experiment code for use with the datasets discussed in paper, originally obtained from
https://github.com/UKPLab/acl2016-convincing-arguments
* python/models -- the implementation of the GPPL method
* python/test -- some simple test scripts for the GPPL methods
* results -- an output directory for storing results

## Requirements

* Python 3
* virtualenv
* The required packages are listed in requirements.txt. You can install them using pip install -r requirements.txt
* Maven -- check if you have the command line program 'mvn' -- required to extract the linguistic features from our experimental datasets. You can skip 
this if you are not re-running our experiments or training a model on UKPConvArg*** datasets.

## How to run the experiments

1. Extract the linguistic features from the data by running:

```
python ./python/analysis/habernal_comparison/run_preprocessing.py. 
```

By default, the data is provided by this repository at ./data and this path is set in ./python/analysis/data_loading.py, line 12.
The data is originally provided by https://github.com/UKPLab/acl2016-convincing-arguments, the copies
are provided here for convenience.

2. Run experiment 1 by running script python/analysis/cycles_demo.py from the root directory of the project:

python ./python/analysis/cycles_demo.py

3. Run experiment 2 (this will take some time):

   ```
   python ./python/analysis/habernal_comparison/scalability_tests.py
   ```

   Generate the plots:

   ```
   python ./python/analysis/habernal_comparison/scalability_plots.py
   ```

   The plots will be saved by default to './documents/pref_learning_for_convincingness/figures/scalability'.

4. Run experiment 3 (this will take some time):

   ```
    python ./python/analysis/habernal_comparison/clean_data_tests.py
   ```

   This script simply sets some parameters for the test:
   * the choice of method
   * dataset
   * features to use with each method

   Given these settings, the experiments are then implemented by ./python/analysis/habernal_comparison/tests.py.

   Compute the performance metrics:

   ```
   python ./python/analysis/habernal_comparison/clean_data_metrics.py
   ```

   This script also just sets some parameters and then calls ./python/analysis/habernal_comparison/compute_metrics.py.

5. Run experiment 4 (this will take some time):

   ```
   python ./python/analysis/habernal_comparison/noisy_data_tests.py
   ```

   Compute the performance metrics:

   ```
   python ./python/analysis/habernal_comparison/noisy_data_metrics.py
   ```

6. Run experiment 5 (this will take some time) for active learning:

   ```
   python ./python/analysis/habernal_comparison/active_learning_tests.py
   ```

   Compute the performance metrics:

   ```
   python ./python/analysis/habernal_comparison/compute_AL_metrics.py
   ```

7. Run analysis of the relevant feature determination:

   ```
   python ./python/analysis/habernal_comparison/features.py
   ```

   The plots will be saved to ./documents/pref_learning_for_convincingness/figures/features2/

8. Produce the output used for error analysis:

   ```
   python ./python/analysis/habernal_comparison/error_analysis.py
   ```

## Template for running on a new dataset with Ling+Glove feature sets

You can use the following script as a template for running GPPL on new datasets 
using the same feature sets as in our paper. If you have another method for
extracting features from your datasets, you may with to skip this example
and look at 'how to use the GPPL implementation'.


```
python ./python/example_use.py
```

The script will train a convincingness model on the UKPConvArgStrict data, then
run it to score arguments in a new dataset. 

Pre-requisite: this script assumes you have carried out step 0 above and 
run "python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features.

## How to use the GPPL implementation

The preference learning method is implemented by the gp_pref_learning class in
python/models/gp_pref_learning.py. 
The template for training and prediction described above contains an example of how
to use the class, but also contains code for extracting our linguistic features and Glove embeddings,
which you may not need.
You can run a simpler example that generates
synthetic data by running python/test/gp_pref_learning_test.py.

The preference learning model and algorithm are implemented by the class 
GPPrefLearning inside python/test/gp_pref_learning_test.py. 
The important methods in this class are listed below; please look at the 
docstrings for these methods for more details:
   * The constructor: set the model hyperparameters
   * fit(): train the model
   * predict(): predict pairwise labels for new pairs
   * predict_f(): predict scores for a set of items given their features, which can be used to rank the items.

## Example usage of GPPL

In this example, we assume that you have a file, 'items.csv', that contains the feature data for some
documents or other items that you wish to model.

Start by loading in some data from a CSV file using numpy:
~~~
item_data = np.genfromtxt('./data/items.csv', dtype=float, delimiter=',', skip_header=1)
item_ids = item_data[:, 0].astype(int) # the first column contains item IDs
item_feats = item_data[:, 1:] # the remaining columns contain item features
~~~

Now, load the pairwise preference data:
~~~
pair_data = np.genfromtxt('./data/pairwise_prefs.csv', dtype=float, delimiter=',', skip_header=1)
# the first column contains IDs of the first items in the pairs. Map these to indices into the item_ids matrix.
items_1_idxs = np.array([np.argwhere(item_ids==iid)[0][0] for iid in pair_data[:, 1].astype(int)])
# the second column contains IDs of the second items in the pairs. Map these to indices into the item_ids matrix.
items_2_idxs = np.array([np.argwhere(item_ids==iid)[0][0] for iid in pair_data[:, 2].astype(int)])
# third column contains preference labels in binary format (1 indicates the first item is preferred, 0 indicates the second item is preferred)
prefs = pair_data[:, 2] 
~~~

Construct a GPPrefLearning object. The following values are reasonable defaults
for the applications we have tried so far:
~~~
from gp_pref_learning import *
from gp_classifier_vb import compute_median_lengthscales # use this function to set sensible values for the lengthscale hyperparameters
model = GPPrefLearning(item_feats.shape[1], shape_s0=2, rate_s0=200, 
                        ls_initial=compute_median_lengthscales(summary_matrix) )
~~~

Now train the object given the data:
~~~
model.fit(items_1_idxs, items_2_idxs, item_feats, prefs, optimize=False)
~~~

Given the fitted model, we can now make predictions about any items given their 
features. These may be new, previously unseen items, or items that were used in 
training. To obtain a score for each item, e.g. to be used for ranking items,
call the following:
~~~
model.predict_f(test_item_feats)
~~~

You can also predict pairwise labels for any items given their features:
~~~
model.predict(test_item_feats, test_items_1_idxs, test_items_2_idxs)
~~~
Here, the test_item_feats object is a matrix where each row is a feature vector
of an item. The test_items_1_idxs and test_items_2_idxs objects are vectors 
(lists or 1-dimensional numpy arrays) containing indices into test_item_feats
of items you wish to compare.

## Setting priors

For setting the prior mean function:
   * when calling fit(), pass in a vector mu0 that is the same size as item_features. Each entry of mu0 is the prior
preference function mean for the corresponding item in item_features
   * when calling predict_f() to predict the score for test items, or 
when calling predict() to predict pairwise labels, 
the argument mu0_output should be used to provide the preference function mean 
for the corresponding items in items_coords, i.e. each item indexed in 
items_coords should have a prior mean value in mu0_output

For the prior precision:
   * The prior precision controls how much the preference learning function can move away from your prior, mu0
   * High precision means that your prior mu0 is very strong, so that when you train the model using fit(), the values
 will not move far from mu0
   * Low precision means that the prior mu0 is very weak and the preference function values estimated using fit() then
 predict_f() will be larger
   * The prior precision itself has a prior distribution, a Gamma distribution with parameters shape_s0 and rate_s0
   * These hyperparameters are equivalent to pre-training the model on n data points with variance v, so you can set them
 as follows...  
   * shape_s0 = n / 2.0
   * rate_s0 = v * n / 2.0
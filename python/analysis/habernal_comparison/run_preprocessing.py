'''
Produce the linguistic features for the experiments on UKPConvArgStrict, UKPConvArgCrowdSample, etc.

The script runs a number of java commands -- in case of problems,
please check the command line output for errors, and check that the output directory contains a file for each of 1052
arguments.

'''
import os
from subprocess import call
import pandas as pd

java_run_path = './acl2016-convincing-arguments/code/argumentation-convincingness-experiments-java/'
java_stanford_path = './acl2016-convincing-arguments/code/de.tudarmstadt.ukp.dkpro.core.stanfordsentiment-gpl/'
mvn_path = './acl2016-convincing-arguments/code/'
classpath = "./target/argumentation-convincingness-experiments-java-1.0-SNAPSHOT.jar:target/lib/*"
stanford_classpath = "./target/de.tudarmstadt.ukp.dkpro.core.stanfordsentiment-gpl-1.7.0.jar:" \
                     "./target/lib/*"


def preprocessing_pipeline(input, output, dataset_name, tmp_data_dir, feature_dir=None, remove_tabs=False):
    # From Ivan Habernal's preprocessing pipeline, first, compile it
    call(['mvn', 'package'], cwd=mvn_path)

    if not os.path.exists(tmp_data_dir):
        os.mkdir(tmp_data_dir)

    # step 0, remove any '_' tokens as these will break the method
    if remove_tabs:
        tmp0 = os.path.abspath('%s/%s0' % (tmp_data_dir, dataset_name))
        if not os.path.exists(tmp0):
            os.mkdir(tmp0)

        for input_file in os.listdir(input):

            if input_file.split('.')[-1] != 'csv':
                continue

            text_data = pd.read_csv(os.path.join(input, input_file), sep='\t', keep_default_na=False, index_col=0)
            text_data.replace('_', ' ', regex=True, inplace=True, )
            text_data.replace('\t', ' ', regex=True, inplace=True, )

            text_data.to_csv(os.path.join(tmp0, input_file), sep='\t')
    else:
        tmp0 = input

    # step 1, convert to CSV:
    tmp = os.path.abspath('%s/%s1' % (tmp_data_dir, dataset_name))
    script = 'PipelineSeparateArguments'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.preprocessing'

    call(['java', '-cp', classpath,
          package + '.' + script,
          tmp0, tmp], cwd=java_run_path)
    print('Completed step 1')

    # step 2, sentiment analysis
    tmp2 = os.path.abspath('%s/%s2' % (tmp_data_dir, dataset_name))
    script = 'StanfordSentimentAnnotator'
    package = 'de.tudarmstadt.ukp.dkpro.core.stanfordsentiment'

    call(['java', '-cp', stanford_classpath,
          package + '.' + script,
          tmp, tmp2], cwd=java_stanford_path)
    print('Completed step 2')

    # step 3, extract features
    tmp3 = os.path.abspath('%s/%s3' % (tmp_data_dir, dataset_name))
    script = 'ExtractFeaturesPipeline'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.features'
    arg = 'false' # not using argument pairs here

    call(['java', '-cp', classpath,
          package + '.' + script,
          tmp2, tmp3, arg], cwd=java_run_path)
    print('Completed step 3')

    # step 4, export to SVMLib format
    script = 'SVMLibEverythingExporter'
    package = 'de.tudarmstadt.ukp.experiments.argumentation.convincingness.svmlib'

    if feature_dir is not None:
        call(['java', '-cp', classpath,
          package + '.' + script, tmp3, output, feature_dir], cwd=java_run_path)
    else:
        call(['java', '-cp', classpath,
          package + '.' + script, tmp3, output], cwd=java_run_path)

    print('Completed step 4')

if __name__ == '__main__':

    # Run this script to run a Java pipeline that produces the libSVM features for the Habernal/Gurevych dataset.

    root_data_dir = os.path.abspath('./data')
    tmp_data_dir = os.path.join(root_data_dir, 'tempdata')

    if not os.path.exists(tmp_data_dir):
        os.mkdir(tmp_data_dir)

    input_dir = os.path.join(root_data_dir, 'argument_data/UKPConvArgStrict-new-CSV')
    output_dir = os.path.join(root_data_dir, 'lingdata')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = os.path.join(output_dir, 'UKPConvArg1-Full-libsvm')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    preprocessing_pipeline(input_dir, output_dir, 'UKPConvArg1-full', tmp_data_dir)

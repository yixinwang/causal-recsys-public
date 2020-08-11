# Reference implementation of the deconfounded recommender

This folder contains the code for _Causal Inference on Recommender
Systems_ (Wang et al., 2020):

+ the empirical study on simulated datasets (Section 3.2)

+ the empirical study on random test sets (Section 3.3)


## Environment

python 2

tensorflow 1.5.0

edward 1.3.5

## How to execute the scripts

### Download the datasets

Download the Yahoo R3 dataset and the coat dataset to `dat/raw/`

+ Yahoo R3: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r

+ Coat (Schnabel et al., 2016): https://www.cs.cornell.edu/~schnabts/mnar/

### Preprocess the datasets

#### Yahoo R3

Run the script `src/preproc/prep_R3_weakgen.py` and
`src/preproc/prep_R3_stronggen.py` to preprocess the dataset.

#### Coat (Schnabel et al., 2016)

Run the script `src/preproc/prep_coat_weakgen.py` and
`src/preproc/prep_coat_stronggen.py` to preprocess the dataset.

#### Simulated datasets

Run the script `src/simdat/simulate_generic.py` to simulate the
datasets.

Run the script `src/preproc/prep_simulate_weakgen.py` and
`src/preproc/prep_simulate_stronggen.py` to preprocess the dataset.

### Perform recommendation

#### Yahoo R3

Run the script `src/causalrec/run_sweep_R3_fitA.sh` and then
`src/causalrec/run_sweep_R3.sh` to perform recommendation.

#### Coat (Schnabel et al., 2016)

Run the script `src/causalrec/run_sweep_coat_fitA.sh` and then
`src/causalrec/run_sweep_coat.sh` to perform recommendation.


#### Simulated datasets

Run the script `src/causalrec/run_sweep_simulation_fitA.sh` and then
`src/causalrec/run_sweep_simulation.sh` to perform recommendation.

### Aggregate results

Run the script `src/causalrec/merge_csv.py` to aggregate results.

## Output

The files `res/*_allres.csv` include output from this implementation.

# References

Y. Wang, D. Liang, L. Charlin, and D.M. Blei. (2020) Causal inference
   on recommender systems. _Proceedings of the 14th ACM Conference on
   Recommender Systems_.

Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak,
Thorsten Joachims (2016). Recommendations as Treatments: Debiasing
Learning and Evaluation. _Proceedings of The International Conference
on Machine Learning (ICML)._
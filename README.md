## Pitt Hopkins Classifier

## About:
an ML classifier that detect patients with a rare disease using the Human Phenotype Onotology (HPO) and simulated genetic dataset. 
The goal is to identify undiagnosed Pitt Hopkins Syndrome (OMIM:610954) patients and select them for free genetic testing.
These patients are extremely rare and there are only a set number of genetic tests that can be given out. 

## Datasets:

### HPO Dataset

The HPO (Human Phenotype Ontology) is a knowledge graph of symptoms. Each symptom node has an HP ID (e.g. `HP:000001`) and name (e.g. "Seizures"), and graph edges to other related symptoms.

Please download the full HPO ontology of symptoms here: https://hpo.jax.org/app/data/ontology. This file is in `.obo` format.

Please download the HPO annotations here: https://hpo.jax.org/app/data/annotations. These annotations connect HPO symptoms with diseases and is used as a ground truth dataset for the model.

### Example Patient Dataset:

Patient data (all simulated) each with symptoms and disease (in this case `OMIM:610954` and `unknown`) annotations.

## Relevant Literature:

1. https://zitniklab.hms.harvard.edu/projects/SHEPHERD/
2. https://arxiv.org/ftp/arxiv/papers/2312/2312.15320.pdf
3. https://www.biorxiv.org/content/10.1101/2022.07.20.500809v1.full.pdf
4. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756558/


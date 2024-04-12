# Toxicity Classifier

This repo is an implementation of a simple CNN classifier for toxic classification.
More information can be found regarding motivation and background [on my personal page](https://allytaylor96.github.io/toxicity-classifier.html#toxicity-classifier).
This readme instead focuses on how to run it from the command line (my natural habitat)!

# Running the Repo

It is designed to be run in three stages: data prep (**prep**), training (**train**) and testing (**test**).

## Setting up the Docker Image

Next, one can run the included `./build_docker_container.sh` command which builds a local Docker image named 'toxicity-classifier'.
After that, you should be able to run `./run_docker_container.sh` which will enter you into the Docker container in an interactive state, while simultaneously mounting the current working directory.

## Setting up configs

Navigate to {repo}/config - you need to update the three configs with the below respectively:
- prep_config.json
    - "dataDir" -> set to where you'd like the interim data files to be stored during prep
- train_config.json
    - "dataDir" -> where the formatted shrunk data files are kept
    - "modelDir" -> where the best model will be stored after training
- test_config.json
    - "dataDir" -> where the formatted shrunk data files are kept
    - "modelDir" -> where the best model is loaded in for inference

## Data Prep

We need to run...

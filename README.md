# Toxicity Classifier

This repo is an implementation of a simple CNN classifier for toxic classification.
More information can be found regarding motivation and background [on my personal page](https://allytaylor96.github.io/toxicity-classifier.html#toxicity-classifier).
This readme instead focuses on how to run it from the command line (my natural habitat)!

# Setting up Docker stuff

One can get started by running the included `./build_docker_container.sh` Bash script which builds a local Docker image named 'toxicity-classifier'.

After that, you should be able to run `./run_docker_container.sh` which will enter you into the Docker container in an interactive state, while simultaneously mounting the current working directory - everything needed for the repo to run is kept and contained here, so when it comes to removing it you can safely `rm -r` the repo.

# Running the Repo

It is designed to be run in three stages: data prep (**prep**), training (**train**) and testing (**test**).

## Setting up the config

Navigate to {repo}/config - you need to update the following field in the toxicity_config.json:
- "repoDir" -> set to your local path of where the repo was cloned to

The rest of the config should be fine with the defaults - the *ftVectorUrl* (which is where the scripts look to download FastText vectors) may need to be updated if FastText change how their vectors are hosted [on their website](https://fasttext.cc/docs/en/english-vectors.html).

One can also change the sentences in *testComments* - these are purely for assessing the model at the end.

## Data Prep



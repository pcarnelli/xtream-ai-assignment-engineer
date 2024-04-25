# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create a Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run

#### Challenge 1

The requested Jupyter notebook, named `challenge1.ipynb`, is located in the directory [notebooks](./notebooks/). There are several options to view it or run it:

1) View in github: The current uploaded version is the result of running the notebook in my local computer and it can be viewed directly from github. (With some caveats, as the `ydata profiling` reports not being displayed. See workaround in notebook).

2) Run in virtual environment: For running the notebook `challenge1.ipynb` the user can build a CONDA virtual environment from the file `environment.yml` or using `venv` (with `Python 3.11.9`) and installing the dependencies from the file `requirements.txt` with `pip`. Both files, `environment.yml` and `requirements.txt`, are also located in the directory [notebooks](./notebooks/).

3) View/run in Google Colab: An already-executed Colab version of the notebook is available [here](https://colab.research.google.com/github/pcarnelli/xtream-ai-assignment-engineer/blob/main/notebooks/challenge1_colab.ipynb). *Note*: a Google account is needed to rerun the notebook.

#### Challenge 2

The automated training pipeline is implemented as a [GitHub Actions](https://docs.github.com/en/actions) workflow. Said workflow is defined in the file `challenge2.yml` located in the directory [.github/workflows](./.github/workflows/).  
The workflow is initiated when a CSV file is pushed to the directory [datasets/diamonds](./datasets/diamonds/) of the repository's main branch. Then, it starts an Ubuntu container and installs the dependencies specified in the file `requirements.txt` located in the repository's root directory. It finishes running the script `train.py` located in the directory [src/steps](./src/steps/). The script `train.py` produces a model object named `pipeline.joblib` and a JSON file named `metrics.json` with train/test metrics, both pushed automatically to the directory [models](./models/).  
Therefore, the steps to run the automated pipeline would be:

1) Commit and push the updated file `diamonds.csv` to the to the directory [datasets/diamonds](./datasets/diamonds/) of the repository's main branch. This action will generate a new commit in the repository.

2) Monitor the workflow status in the *Actions* tab of the repository at github.com (e.g. [https://github.com/pcarnelli/xtream-ai-assignment-engineer/actions](https://github.com/pcarnelli/xtream-ai-assignment-engineer/actions)).

3) If the workflow ends successfully, the files `pipeline.joblib` and `metrics.json` will be pushed and a new commit will be added to the repository's history. The commit's description will be "feat: update model after training workflow" and it will be co-authored by github-actions[bot].

Please, refer to the docstrings in the scripts for more information.  
If issues arise during local testing of the script `train.py`, in the directory [docker/challenge2](./docker/challenge2/) there is a Dockerfile for building an image with the needed requirements. Run instructions can be found in the file `README.md` located in the same directory.

#### Challenge 3

The REST API is built with [Flask](https://flask.palletsprojects.com/) in the script `app.py` located in the directory [src/](./src/). It can be started from the project's root directory with the following CLI command:

    $ python src/app.py

The script `request.py`, located in the directory [src/](./src/), can be used to test the REST API locally. This script can be executed from the project's root directory with the following CLI command:

    $ python src/request.py

The request contains a JSON data payload (read from file `res/payload.json`) with the following fields: 'carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity'. The payload can contain more than one item. The response is expected to be in JSON format with a field 'price' that contains a list with the prediction/s. If the request is successful (code 200), the response is printed and saved to disk (file `res/prediction.json`).  
Please, refer to the docstrings in the scripts for more information.  
If issues arise during testing of the REST API, in the directory [docker/challenge3](./docker/challenge3/) there is a Dockerfile for building an image with the needed requirements. Run instructions can be found in the file `README.md` located in the same directory.

### Challenge 4



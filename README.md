# QA_engine_nlp
Project's Github: [here](https://github.com/mvonwyl/epita/tree/master/NLP/08)

In order to get the project running and test the code please proceed with the following:
* Create a python3 environment, for exemple: 
```
python3 -m venv sentiment_analysis_env
```
  and activate it:
  ```
  . sentiment_analysis_env/bin/activate
  ```
 * Install the dependincies from the requirements.txt file:
```
pip3 install -r requirements.txt
```
* Next open the directory with jupyter:
```
jupyter notebook
```
* Cells already have their outputs shown, but feel free to run all cells again (order of cell execution very important).

When running the notebooks, be sure to have the files utils.py and util_index.py either in the same level as the notebooks if running them locally, or loaded in Colab.
We recommend running our notebooks in Colab and take advantage of Colab's GPUs.

Regarding our project structure:

* `utils.py` includes functions for tokenization, processing, prediction etc.

* `utils_addcontext.py` includes functions to add unique DBPedia contexts into our dataset

* `util_index.py` includes functions for using and testing the different pre-trained models, the MRR computation and some functions to get results from these models (like getting the top k relevant documents regarding a question)

* `question_answering_student.ipynb` includes the code to train our question-answering model, based on the `question_answering.ipyng` notebook 

* `NLP_Final_Part2.ipynb` contains all the code to create our searchable index: the preprocessing to increase the volume of our dataset, testing the different models' performances etc.

* `Everything_together_3.ipynb` contains the combination of `question_answering.ipynb` and `NLP_Final_Part2.ipynb`, which is our trained model from `question_answering.ipynb` being able to predict which document is the most relevant one from 10 documents provided by the Nearest Neighbors approximation in `NLP_Final_Part2.ipynb`.

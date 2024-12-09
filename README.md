# 👩‍⚖️ Judge a Book by its Cover 📕

Code for reproducing experiments in the paper "Judge a Book by its Cover: Investigating Multi-modal LLMs for Multi-page Handwritten Document Transcription", submitted to the [DocUI@AAAI25 workshop](https://sites.google.com/view/docui-aaai25).

### Installation
For Unix:
```
conda create --name judge python=3.10
conda activate judge
pip install -r requirements.txt
pip install -e .
mkdir data plots results data
```

You will also have to set up the AWS CLI and API keys for OpenAI and OCR engines if you haven't done so already (see `.example.env` for an example `.env` file).

### Data
Experiments are based on the [IAM Handwriting DB](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), which should be downloaded and unzipped inside `/data` before running `notebooks/00_iam_preprocessing.py`. 

(Registration to access the dataset can be faulty, I found [these instructions](https://www.reddit.com/r/datasets/comments/l2agom/comment/ksww8co/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) to be convoluted but effective.)


### Running Experiments
Experimental results can be reproduced with the notebooks below as follows:
- `notebooks/00_iam_preprocessing.py`: run OCR engines on the IAM dataset, save to pandas dataframes, produce multi-page datasets and save dataframes for downstream notebooks to .pkl files
- `notebooks/01_experiments.py`: make OpenAI API calls with various prompting strategies to get improved OCR transcriptions
  - Can be run as a notebook or using command line arguments with .yaml config files in `configs/`
  - Command line commands for running all experiments e2e are in `notebooks/run_notebooks.sh`
- `notebooks/01_submit_save_batch_calls.py`: optional notebook to make batch API calls
  - By including `batch_api: True` in the config file for `01_experiments`, API calls are written to a batch submission file  in `data/batch_api`. 
  - With this notebook, batch calls can be submitted to the API, the outputs can be downloaded, and then written to the GPT cache file. Then you should be able to re-run `01_experiments` and the cached outputs will load. Make sure to run one cell at a time and only follow the instructions accompanying each cell
- `notebooks/02_error_catching.py`: catches simple LLM errors and updates outputs accordingly, as described in ther 'Error catching' sub-section of the paper.
- `03_{performance_tables, dropfirst_table, pagecount_plot, pareto_plot}.py`: uses output files from notebooks `01`/`02` to produce Tables 1,2 and Figures 1,2 in the paper

**N.B.** though we use a seed, OpenAI API calls are [not strictly reproductible](https://platform.openai.com/docs/advanced-usage#reproducible-outputs), so slight differences in output are possible.

To avoid the cost incurred by running all experiments from scratch, the cache files for these experiments can be downloaded [here](CACHE FILES TBC) and placed inside `/data`
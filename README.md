# MCQ Quiz Creator Application 


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n llmapp python=3.8 -y
```

```bash
conda activate llmapp
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your OpenAI_API_KEY as follows:

```ini
OpenAI_API_KEY= "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# Finally run the following command
streamlit run app.py
```

Now,
```bash
open up : http://localhost:8501
```


### Techstack Used:

- Python
- LangChain
- Streamlit 
- OpenAI
- GPT 3.5 Turbo
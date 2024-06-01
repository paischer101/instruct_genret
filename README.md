# GenRec

Code library for generative recommendation models.

To train a model using specific config:

``` bash
python run.py --config=path/to/config
```

Data will be downloaded automatically, and RQ-VAE will also be trained and id will be generated and stored in ./ID_generation/ID before TIGER training starts.

Download pretrained model checkpoints and id assignment from [here](https://drive.google.com/drive/folders/1GmaSlna0smM0wJSbA-xddUT2uIk52aG6?usp=sharing)

check this [google sheet](https://docs.google.com/spreadsheets/d/1qAsxAA9qTfETum0dcF4OiX4NYbPYzJWE__DY7NPXEPI/edit?usp=sharing) for current experiment results. The model files are results from experiment id 1,3,5, check google sheet for detailed experiment setups.

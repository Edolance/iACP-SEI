## iACP-SEI

In this study, we developed a new method called iACP-SEI for the identification of anticancer peptides based on the protein language model ESM2 and the stacked ensemble learning model.

This repository contains the source code, the data and link of the pretrained embedding models accompanying the paper—iACP-SEI: An anticancer peptide identification method incorporating sequence evolutionary information.

## Installation instructions

**Step 1: Create a virtual environment**

```shell
conda create -n iACP-SEI python=3.8
```

**Step 2: Activate the environment**

```shell
conda activate iACP-SEI
```

**Step 3: Install the required packages**

```shell
pip install -r requirements.txt
```

**Step 4: Run the script**

```shell
python iACP-SEI.py
```

If the following output is displayed, the installation was successful.

<p align="center">
  <img src="./figures/initial-interface.png" alt="initial-interface" width="800">
</p>

**Step 5: Input requirements**

The input should be a peptide sequence in FASTA format. Use the `-m` option to select different models:

- `Alt`: Model trained on the Alt-Dataset.
- `Main`: Model trained on the Main-Dataset.
- `Merged`: Model trained on the Merged-Dataset.

**Additional requirements**

You need to download the ESM models into the `esm_model` folder. Download it from:

[http://public.aibiochem.net/peptides/iACP-SEI/esm_model/](http://public.aibiochem.net/peptides/iACP-SEI/esm_model/)


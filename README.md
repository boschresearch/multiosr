# Multi-Attribute Open Set Recognition

Open Set Recognition (OSR) extends image classification to an open-world setting, by simultaneously classifying known classes and identifying unknown ones. While conventional OSR approaches can detect Out-of-Distribution (OOD) samples, they cannot provide explanations indicating which underlying visual attribute(s) (e.g., shape, color or background) cause a specific sample to be unknown. In this work, we introduce a novel problem setup that generalizes conventional OSR to a multi-attribute setting, where multiple visual attributes are simultaneously predicted. Here, OOD samples can not only be identified but also categorized by their unknown attribute(s). We propose simple extensions of common OSR baselines to handle this novel scenario. We show that these baselines are vulnerable to shortcuts when spurious correlations exist in the training dataset. This leads to poor OOD performance which, according to our experiments, it is mainly due to unintended cross-attribute correlations of the predicted confidence scores. We provide an empirical evidence showing that this behavior is consistent across different baselines on both synthetic and real world datasets.

For more information about this work, please read our GCPR 2022 paper:

> Saranrittichai, P., Mummadi, C., Blaiotta, C., Munoz, M., & Fischer, V. (2022). Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain. In Proceedings of the German Conference on Pattern Recognition (GCPR).

## Table of Contents
- [Installation](#installation)
- [Run Studies](#run-studies)
- [Questions and Reference](#questions-and-reference)

## Installation

1. First we recommend to setup a python environment using the provided `environment.yml` and install the package:

```
conda env create -f environment.yml
source activate sourcegen
pip install -e .
```

2. Navigate to `data/diagvibsix` and follow the instruction on `diagvib_setup_instruction.txt` to prepare data for the DiagViB-6 framework. In this work, we customize DiagViB-6 for our use cases. Official DiagViB-6 release can be found [here](https://github.com/boschresearch/diagvib-6).

3. Navigate to `data/ut_zappos` and follow the instruction on `ut_zappos_setup_instruction.txt` to prepare data and splits for the UT-Zappos dataset.

## Run Studies

We provide python scripts to run all studies. Lists of methods, datasets and training seeds can be customized in the file `multiosr/studies/run_studies`. Since multiple jobs will be executed, it is suggested to modify the execution of `command_str` in the line 42 to be executed on your GPU clusters in parallel. The command to run studies is:
```
python -m multiosr.studies.run_studies
```

## Questions and Reference
Please contact [Piyapat Saranrittichai](mailto:piyapat.saranrittichai@de.bosch.com?subject=[GitHub]%20SourceGen)
or [Volker Fischer](mailto:volker.fischer@de.bosch.com?subject=[GitHub]%20SourceGen) with
any questions about our work and reference it, if it benefits your research:
```
@InProceedings{
    Saranrittichai_2022_GCPR,
    author = {Saranrittichai, Piyapat and Mummadi, Chaithanya Kumar and Blaiotta, Claudia and Munoz, Mauricio and Fischer, Volker},
    title = {Multi-Attribute Open Set Recognition},
    booktitle = {German Conference on Pattern Recognition (GCPR)},
    month = {September},
    year = {2022}
}

# Non-small-cell lung cancer classification via RNA-Seq and histology imaging probability fusion

### *Francisco Carrillo-Perez, Juan Carlos Morales, Daniel Castillo-Secilla, Yésica Molina-Castro, Alberto Guillén, Ignacio Rojas & Luis Javier Herrera*

Implementation of the BMC Bioinformatics journal paper [Non-small-cell lung cancer classification via RNA-Seq and histology imaging probability fusion](https://rdcu.be/cyE5N)

---
## Abstract

### Background

Adenocarcinoma and squamous cell carcinoma are the two most prevalent lung cancer types, and their distinction requires different screenings, such as the visual inspection of histology slides by an expert pathologist, the analysis of gene expression or computer tomography scans, among others. In recent years, there has been an increasing gathering of biological data for decision support systems in the diagnosis (e.g. histology imaging, next-generation sequencing technologies data, clinical information, etc.). Using all these sources to design integrative classification approaches may improve the final diagnosis of a patient, in the same way that doctors can use multiple types of screenings to reach a final decision on the diagnosis. In this work, we present a late fusion classification model using histology and RNA-Seq data for adenocarcinoma, squamous-cell carcinoma and healthy lung tissue.

### Results

The classification model improves results over using each source of information separately, being able to reduce the diagnosis error rate up to a 64% over the isolate histology classifier and a 24% over the isolate gene expression classifier, reaching a mean F1-Score of 95.19% and a mean AUC of 0.991.

### Conclusions

These findings suggest that a classification model using a late fusion methodology can considerably help clinicians in the diagnosis between the aforementioned lung cancer cancer subtypes over using each source of information separately. This approach can also be applied to any cancer type or disease with heterogeneous sources of information.

 
---

## Implementation

Code can be found in the src folder:

- **[gen_train.py]** -- train the rna-seq model
- **[wsi_train.py]** -- train the wsi model
- **[train_pred.py]** -- utilities for wsi training
- **[data_reader.py]** -- datareader class for reading data
- **[dataset.py]** -- pytorch dataset extension for the dataset
- **[utils.py]** -- different utilities functions
- **[obtain_patches.py]** -- obtain patches from WSI images
- **[read_img.py,read_gen.py]** -- utility reader for wsi and rna-seq data
- **[late_fusion.py]** -- perform the late fusion over the probabilities


---

## Citation

If you find this work useful, please cite as follows:

```
Carrillo-Perez, F., Morales, J.C., Castillo-Secilla, D. et al. Non-small-cell lung cancer classification via RNA-Seq and histology imaging probability fusion. BMC Bioinformatics 22, 454 (2021). https://doi.org/10.1186/s12859-021-04376-1
```

```
@article{carrillo2021non,
  title={Non-small-cell lung cancer classification via RNA-Seq and histology imaging probability fusion},
  author={Carrillo-Perez, Francisco and Morales, Juan Carlos and Castillo-Secilla, Daniel and Molina-Castro, Y{\'e}sica and Guill{\'e}n, Alberto and Rojas, Ignacio and Herrera, Luis Javier},
  journal={BMC bioinformatics},
  volume={22},
  number={1},
  pages={1--19},
  year={2021},
  publisher={BioMed Central}
}
```

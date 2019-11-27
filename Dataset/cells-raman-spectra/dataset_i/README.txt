This dataset was collected in the University of Chemistry and Technology, Prague during work on cancer detection.
It contains Raman spectra of the culture medium, corresponding to several kinds of cancer and normal cells.
Dataset consists of 12 folders with 3 CSV files in each. Folders are named after specific samples (tabulated
below). Each CSV in folder contains spectra of medium, collected on the gold nanourchins functionalized with
corresponding moiety. Please refer to the original publication for details.

Corresponding Raman shift scale (x-values) in reciprocal centimeters: 
start: 100, end: 4278, number of points: 2090
It is convinient to use NumPy or MATLAB linspace function for it:
> x = linspace(100, 4278, 2090)

+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| Label of the sample | cells | characterization                       | origin          | Serum content | No of samples |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| A                   | A2058 | melanoma cells                         | Cell line       | 10.00%        | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| A-S                 | A2058 | melanoma cells                         | Cell line       | 0.00%         | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| G                   | G361  | melanoma cells                         | Cell line       | 10.00%        | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| G-S                 | G361  | melanoma cells                         | Cell line       | 0.00%         | 8             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| HPM                 | HPM   | neonatal highly pigmented  melanocytes | Cell line       | 10.00%        | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| HPM-S               | HPM   | neonatal highly pigmented  melanocytes | Cell line       | 0.00%         | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| HF                  | HF    | normal skin fibroblasts                | Primary culture | 10.00%        | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| HF-S                | HF    | normal skin fibroblasts                | Primary culture | 0.00%         | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| ZAM                 | ZAM   | tumour associated fibroblasts          | Primary culture | 10.00%        | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| ZAM-S               | ZAM   | tumour associated fibroblasts          | Primary culture | 0.00%         | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| DMEM                | None  | pure medium - control                  |                 | 10.00%        | 8             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+
| DMEM-S              | None  | pure medium - control                  |                 | 0.00%         | 9             |
+---------------------+-------+----------------------------------------+-----------------+---------------+---------------+

Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License.
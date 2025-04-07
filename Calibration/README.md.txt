<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-04-07 16:10:05
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-04-07 21:02:50
 * @FilePath: /ntu/gp_uwb/GP-UWB-calibration/Calibration/README.md.txt
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

# Train and test the GP model

### 1. MCD-NTU Dataset
The Mcd-NTU dataset was collected by an ATV moving within a campus environment, where 10 UWB base stations (numbered 0 to 9) were distributed across the area. As shown in the MCDUWB.zip file (which can be extracted to the current folder), all data has been transformed into a unified coordinate system. Therefore, coordinate system transformations do not need to be considered when fitting the GP kernel.

It should be noted that as of April 2025, only 6 sequences from the MCD-NTU dataset have been open-sourced. However, more sequences were actually collected during data acquisition. We used these unpublished sequences to fit our kernel. These additional sequences will be released in the future.

### 2. Fit the GP kernel using fit3D_MCD.py
After placing the data in the current folder, modify `save_dir` and `train_data_list` in `fit3D_MCD.py`, then run:

```bash
python fit3D_MCD.py
```

### 3. Sample the maxest point in the guessing point
To accelerate the search for extremum points (i.e., UWB base station locations), we first use sphere fitting to estimate rough positions of the UWB base stations and then sample around these locations to find extremum points. Alternatively, you can start from scratch and perform a broader search for maxima, though this may require more iterations.Then run:

```bash
python draw3D_arround_MCD.py
```

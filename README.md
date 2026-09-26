# Opal glass method for microalgae pigment quantification, CNN algorithm

Repository hosting the Python codes (please be kind with the typos in the comments) and data associated with the articles:
- [Opal glass method, an extraction-free way to access microalgae pigment content.](https://doi.org/) Pozzobon, V., Arnoudts, C., & Levasseur, W. (2026). _Bioresource Technology_, XXX, XXX. [(Publisher Open Access)](https://pdf.sciencedirectassets.com) [(PDF file)](https://victorpozzobon.github.io/assets/preprints/Pozzobon_2026_f.pdf) [(Supplementary materials)](https://victorpozzobon.github.io/assets/preprints/Pozzobon_2026_f_Supplementary_Materials.pdf)

It has been tested successfully on October 2026.

## Structure

__Step 1: train the CNN__

By executing _CNN_Train_and_plot_, you shall be able to train a CNN predicting cell chlorophyll a content, in the first section of the script. As you will see, the file also hosts the CNN structure. To run it, you will need _tensorflow_, _keras_, _cuda_ (not strictly necessary, but speeds up the training a lot), _scikit-learn_, and _matplolib_ (to print out the results). Please note that the script is intentionally very barebone/straight to the point (e.g., the cross-validation only picks the last fold to avoid too long training). It is meant to ease understanding and reuse. So feel free to adapt it. 

__Step 2: plot the results__

You can plot the data with your favorite framework/software. Here, the graph compares cell chlorophyll a content on a biological run that was not included in the training set. 

![Image not found](./Run1_CNN.png?raw=true)

## Contact

Please feel free to contact me. You can find my details on: https://victorpozzobon.github.io/

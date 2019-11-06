## Log on development
* [X] Finish initial segmentation with the origin model;
* [ ] Add time points as separate channel to control cell size;
    * :x: Doesn't work as a separate input channel;
    * --> In coporate time with MLP and **reshape and concatenate the time features as a channel layer**.
    [Temporal consistency objectives regularize the learning of disentangled representations](https://arxiv.org/abs/1908.11330?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)
* [ ] How to constrain the width of the predicted membrane mask to get more precise results; 
* [x] Use nucleus voroni segmentation as as constrain --> Distanc between *membrane* and *voroni*
might mislead the network. Combine nucleus points as a separate **binary** channel (large error
 when combine as Gaussian distance map, especialy when t>100). 
* [ ] Loss function on rays distance from the center to the membrane surface --> **PolarIoU loss**
[PolarMask Single shot instance segmentation with polar representation](https://www.evernote.com/l/AoJhyTxvDUpHB4W3c0I_mFJ53vH6_kb0zYQ/);
* [ ] Augmentation on weak boundary --> [A weakly supervised method for instance segmentation of biological cells](https://www.evernote.com/l/AoJWPvj9qcJNBa3uJVKN3qZZpMBn1q210QY/);
* [ ] Few-shot training with only a few slices annotation cross the nucleus center. For each
cell we only annotate two vertical plans cross the nucleus. Then use distance constrained 
learning to learn the **overall volumetric distance transformation** and **further 
postprocessing**. Only loss at the **annotated** and **background away from the embryo** will
be counted. 
* [ ] Show let network choose **which slices can make more contribution** and **aollow different
weights on neighboring slices** 
* [ ] Embryo outside <0, inside >0, membrane =0.
* [ ] For <u>*boundary pixels*</u>, the distance to the centers of neighboring pixels should be
embeded with smooth changes. From the prediction results, pixels from the same class **is grouped
together**. So only boundary pixels are needed to be counted into the distance constrain.
* [x] The **weight at distance wieghts should not =0** when the region overlaps with the target.
While cross entropy loss only counts on the *p* value at the index, distance aware also consider 
the prediction far away th target. 
* [ ] In regression, neighboring pixels also needs to be spacialy constrained;
* [x] Imply adaptive regression loss in MSE. [Efficient and robust cell detection a structed 
regression approach](https://www.evernote.com/l/AoICCQ4edg5Edbyb85gHuXHpnlAlDYL_rg0/);
* [ ] Although **receptive field** and and **high resolution information** are emphasized, they
 are only solved with original concatenation.  [Efficient and robust cell detection a structed 
regression approach](https://www.evernote.com/l/AoICCQ4edg5Edbyb85gHuXHpnlAlDYL_rg0/)
* [ ] Utilize different kinds of distance map. [Segmentation of nuclei in ... by regression of 
distance map](https://www.evernote.com/l/AoLGy88mbcdGn78jHUYnsAXsRHo-klBkOf0/)

## TODO in experiment
* [ ] Compare *binary*, *multiclass* classification and *regression*. 
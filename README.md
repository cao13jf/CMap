## Log on development
* [X] Finish initial segmentation with the origin model;
* [ ] Add time points as seperate channel to control cell size;
* [ ] How to constrain the width of the predicted membrane mask to get more precise results; 
* [ ] Use nucleus voroni segmentation as as constrain;
* [ ] Loss function on rays distance from the center to the membrane surface --> universe;
* [ ] Augmentation on weak boundary;
* [ ] Few-shot training with only a few slices annotation cross the nucleus center. For each
cell we only annotate two vertical plans cross the nucleus. Then use distance constrained 
learning to learn the **overall volumetric distance transformation** and **further 
postprocessing**. Only loss at the **annotated** and **background away from the embryo** will
be counted. 
* [ ] Show let network choose **which slices can make more contribution** and **aollow different
weights on neighboring slices** 
* [ ] For <u>*boundary pixels*</u>, the distance to the centers of neighboring pixels should be
embeded with smooth changes. From the prediction results, pixels from the same class **is grouped
together**. So only boundary pixels are needed to be counted into the distance constrain. 

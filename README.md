Oarfish
=======
Oarfish is an experiment to see how well deep learning works at classifying degree resolution
all sky images from the [Orville Wideband Imager](https://github.com/lwa-project/orville_wideband_imager)
running at the LWA stations.

Approach
--------
Oarfish uses a two stage classifier to assign a quality score and label to an all sky image.  The
first stage is a binary classifier that is used to distinguish "good" data (data with low to moderate
RFI that is free of instrumental problems) from "bad".  The second stage takes this a step farther
and tries to distinguish between:
 * good images,
 * medium RFI images - images where the brightest is comparable to that of "A team" soures,
 * high RFI images - where the RFI is brigher than anything else in the sky,
 * corrupted images that are likely caused by instrumental problems,
 * images where the Sun is flaring, and
 * images where Jupiter is bursting.

The first stage is good for separating images that can generally be used for science from those that
cannot but it cannot be used to determine what the RFI environment is like beyond good vs. bad.  The
second stage is good for understanding *why* an image is bad but it also has trouble with separating
no RFI from low RFI from medium RFI.

To overcome this oarfish introduces a "quality score" which combines the two classifier results into
a single number.  It take the binary classification confidence (downweighted by 50%) and combines it
with the top two (by confidence) results from the multi-class stage to create a single number that
ranges from 0 (bad) to 1 (good).  This number is then mapped to a text label that is one of:
 * good
 * low RFI
 * medium RFI
 * high RFI
 * bad

The division of quality score into bad to good is not uniform and the thresholds between the levels
have been adjusted to match what a person might label a variety of images.

Training
--------
The two models included with the library are trained using a collection of images pulled from the [LWA-SV Orville archive](https://lda10g.alliance.unm.edu/Orville/)[^1].
These images span the full range of time and frequency that Orville has been operating over at
Sevilleta.  For the binary classifier a training set of roughly 900 good and 900 bad images was used.
The multi-class model was trained on:
 * roughly 2000 RFI-free images,
 * about 1400 low to medium RFI images,
 * 700 high RFI images,
 * a little over 900 corrupted images,
 * almost 140 images where the Sun was flaring, and
 * almost 40 images where Jupiter was bursting.

The validation sets for both followed a 80/20 split for training/validation.

For processing the The Stokes I and |V| images were normalized to 0 to the 99.75-th percentile of the
Stokes I image and resampled to a uniform size 256 by 256 pixels for pattern recognition.  In addition,
key features were extracted from the images that traced the A team, the Sun/Jupiter, and characterized
the horizon and sky contrast.

Using
-----
The easiest way to use oarfish is through the `oarfish.data.MultiChannelDataset` and `oarfish.predict.DualModelPredictor`
classes.  `MultiChannelDataSet` is a PyToach `DataSet` sub-class that prepares a collection of NumPy
array for pattern recognition and astronomical feature extraction.  `DualModelPredictor` wraps the model
loading a prediction into a single object.  It loads in the binary and multi-class models, runs the
prediction on a `DataSet`, and returns a full set of metrics.

If you are working with Orville .oims files there is also a ZeroMQ-based server and client included with
the scripts.

[^1]: There were a handful of LWA-NA Orville images used but not enough to dramatically alter the training.

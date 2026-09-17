Oarfish Training Data
=====================
Collection of labelled all-sky images used to train the models shipped with
[oarfish](https://github.com/lwa-project/oarfish).  The images come predominately
from the [LWA1 PASI archive](https://lda10g.alliance.unm.edu/PASI/) and the
[LWA-SV Orville archive](https://lda10g.alliance.unm.edu/Orville/), with a small
number of images from the Orville imager at LWA-NA.  The images span the full range of
time and frequency those imagers have operated over.  Every image was labelled by hand.

Layout
------
```
manifest.csv
DATASET.md                 (this file)
LICENSE                    (CC BY 4.0)
<class>/<split>/<file>.npz
```

`<class>` is one of the seven multi-class labels and `<split>` is `train` or `val`,
matching how the shipped models were trained and validated (roughly a 75/25 split):

| class        | train | val  | description |
|--------------|------:|-----:|-------------|
| `good`       | 2166  | 704  | RFI-free or nearly so; usable for science |
| `medium_rfi` | 1644  | 413  | RFI comparable in brightness to the "A team" sources |
| `high_rfi`   | 972   | 243  | RFI brighter than anything else in the sky |
| `corrupted`  | 982   | 247  | Instrumental problems: missing data, bad calibration, etc. |
| `sun`        | 341   | 87   | A flaring Sun dominates the image |
| `jupiter`    | 194   | 50   | A Jovian burst dominates the image |
| `lightning`  | 344   | 59   | Lightning within range of the station (usually spatially resolved) |

File Names
----------
Each `.npz` is named after the image archive file it was extracted from, plus an integration
index, e.g.

```
59192_59192_090000_31.700MHz_48.200MHz.oims_0.npz     Orville: <MJD>_<MJD>_<HHMMSS>_<band>.oims_<integration>
57265:16_38.100MHz_5.0s.pims_536_0.npz                PASI:    <MJD>:<hour>_<freq>_<int>.pims_<integration>_<channel>
```

**Note: File names are not unique.**  An Orville name carries the file's overall band but
not the channel, so two channels of the same integration collapse to the same name.  These
may land in different splits or different classes, so the `<class>/<split>/<file>` path is
unique;  use `start_freq_mhz` in the manifest to tell the collided pairs apart.  A handful
of images are byte-for-byte duplicates of another file in the set (the `duplicate_of` column).

File Contents
-------------
Each file is a NumPy `.npz` archive with two entries:

 * `data` -- `float32` array of shape `(nchan, 4, npix, npix)`: `nchan` frequency channels
   (1, 2, or 6), four Stokes parameters in the order I, Q, U, V, and a square image of `npix`
   pixels on a side (64 to 240, most commonly 128).  The images are in the imager's native
   orthographic projection with the zenith at/near the center; the pixel size is in the
   metadata.  Values are in the imager's uncalibrated units and have **not** been normalised.
 * `info` -- a pickled Python `dict` of the image header.  Loading it requires
   `numpy.load(path, allow_pickle=True)` and then `info = f['info'].item()`.

The header keys in the metadata vary with the imager and its vintage.  Keys that are always
present:

| key             | meaning |
|-----------------|---------|
| `start_time`    | Start of the integration, MJD (UTC) |
| `int_len`       | Integration length; in days for most files, in seconds for a few hundred Orville files |
| `lst`           | Local sidereal time at the start; hours for Orville, days for PASI |
| `start_freq`    | Center frequency of the first channel in Hz |
| `bandwidth`     | Channel bandwidth in Hz |
| `pixel_size`    | Degrees per pixel at the phase center |
| `center_ra`, `center_dec` | Phase/pointing center (~zenith) in degrees |
| `stokes_params` | `'I,Q,U,V'` |
| `fill`          | Fraction of the visibilities present |

Orville files also carry information about the phase center's topocentric position (`center_az`
and `center_alt`), the ASP filter and attenuator settings (`asp_filter`, `asp_atten_*`), and sometimes `weighting`, `ngrid`, and `station`.  PASI files carry `station`, `freq`, `gain`, and
`visFileName` instead.  The manifest normalises the useful ones so that none of this needs
unpickling to find an image.

Manifest
--------
`manifest.csv` contains one row per image:

| column           | meaning |
|------------------|---------|
| `path`           | `<class>/<split>/<file>` relative to this directory |
| `class`          | One of the seven classes above |
| `split`          | `train` or `val` |
| `md5`            | Checksum of the file |
| `bytes`          | File size |
| `nchan`, `npix`  | Shape of `data` |
| `station`        | `LWA1`, `LWASV`, or `LWANA` where the header says or the file type implies it; blank otherwise |
| `start_mjd`      | Start of the integration, MJD (UTC) |
| `lst_hours`      | Local sidereal time in hours |
| `start_freq_mhz` | Center frequency of the first channel in MHz |
| `bandwidth_mhz`  | Channel bandwidth in MHz |
| `int_len_s`      | Integration length in seconds (~5 s throughout) |
| `source`         | The archive file the image was extracted from |
| `duplicate_of`   | Path of an earlier row with identical content, if any |

To read in an image with `oarfish` use:

```python
from oarfish.data import LWATVDataset
ds = LWATVDataset(['jupiter/train/59200_59200_040000_31.700MHz_48.200MHz.oims_3.npz'])
```

This can also be done directly with NumPy via:

```python
import numpy as np
f = np.load(path, allow_pickle=True)
data, info = f['data'], f['info'].item()
stokes_i = data[:, 0]        # (nchan, npix, npix)
```

Provenance and Caveats
----------------------
 * Labels are the judgement of one person looking at the Stokes I and |V| images.  The rare
   classes (`sun`, `jupiter`, `lightning`) were curated to be clear-cut examples on purpose.
 * The images are not evenly distributed in time, frequency, or station; LWA-SV Orville data
   dominate, and the PASI (LWA1) images are mostly at 38.1 MHz.
 * There is no flux density calibration.  Pixel values are comparable within an image, not
   between images.

License
-------
The images and their labels are released under the
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
license (CC BY 4.0; full text in `LICENSE-DATA`).  If you use them, please cite this dataset:

> Dowell, J., *Oarfish Training Data*, Long Wavelength Array, 2026, https://fornax.phys.unm.edu/lwa/data/oarfish/

and include the following in your acknowledgements:

> Construction of the LWA has been supported by the Office of Naval Research under Contract
> N00014-07-C-0147.  Support for operations and continuing development of the LWA1 is provided
> by the Air Force Research Laboratory and the National Science Foundation under grants
> AST-2107845.  This work was sponsored in part by the Air Force Office of Scientific Research
> (AFOSR) Lab Task 23RVCOR002.

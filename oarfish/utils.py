import numpy as np
from scipy.ndimage import gaussian_filter
from functools import lru_cache
from typing import Tuple, Union, List, Dict, Optional

from astropy.coordinates import SkyCoord, get_body, EarthLocation
from astropy.time import Time
from astropy.wcs import WCS


HORIZON_ALT_DEG = 30.0
HORIZON_ZA_DEG = 90.0 - HORIZON_ALT_DEG



SOURCES = {'CygA': SkyCoord('19h59m28.356s', '40d44m02.10s', frame='icrs'),
           'CasA': SkyCoord('23h23m27.951s', '58d48m42.40s', frame='icrs'),
           'VirA': SkyCoord('12h30m49.423s', '12d23m28.04s', frame='icrs'),
           'TauA': SkyCoord( '5h34m31.946s', '22d00m52.15s', frame='icrs')
          }


EXT_SOURCES = {'HerA': SkyCoord('16h51m07.989s',  '4d59m35.55s', frame='icrs'),
               'HydA': SkyCoord( '9h18m05.668s', '-12d05m43.81s', frame='icrs'),
               'ForA': SkyCoord( '3h22m41.788s', '-37d12m29.52s', frame='icrs')
              }


@lru_cache(maxsize=32)
def _topo_wcs_to_altaz(xsize: int, ysize: int, topo_wcs: WCS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to cache successive WCS lookups to get the azimuth and
    altitude of each pixel in the image (in degrees).
    """
    
    x, y = np.arange(xsize), np.arange(ysize)
    x, y = np.meshgrid(x, y)
    
    sc = topo_wcs.pixel_to_world(y, x)
    az = (270 - sc.ra.deg) % 360
    alt = sc.dec.deg
    
    return az, alt


@lru_cache(maxsize=32)
def _wcs_to_skycoord(xsize: int, ysize: int, wcs: WCS) -> SkyCoord:
    """
    Helper function to cache successive WCS lookups to get the RA and dec
    of each pixel in the image (in degrees).
    """
    
    x, y = np.arange(xsize), np.arange(ysize)
    x, y = np.meshgrid(x, y)

    return wcs.pixel_to_world(y, x)


def extract_sky(stokes_i: np.ndarray, stokes_v: np.ndarray, topo_wcs: WCS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the 1D numpy.ndarray with the sky values from the Stokes I and |V|
    images.
    """
    
    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_i.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    az, alt = _topo_wcs_to_altaz(xsize, ysize, topo_wcs)
    
    sky = np.where(alt > HORIZON_ALT_DEG)
    sky_az = az[sky].ravel()
    order = np.argsort(sky_az)
    
    sky_i = np.zeros((nchan, len(sky[0])), dtype=stokes_i.dtype)
    sky_v = np.zeros((nchan, len(sky[0])), dtype=stokes_v.dtype)
    for c in range(nchan):
        sky_i[c] = stokes_i[c][sky].ravel()[order]
        sky_v[c] = stokes_v[c][sky].ravel()[order]
        
    if reshape_needed:
        sky_i = sky_i[0]
        sky_v = sky_v[0]
        
    return sky_i, sky_v


def characterize_sky(sky_i: np.ndarray, sky_v: np.ndarray) -> Union[Dict[str, float],
                                                                    List[Dict[str, float]]]:
    """
    Given the extracted Stokes I and |V| sky, compute a few metrics
    to characterize it.  They are:
     * Median flux in Stokes I
     * Inter-quartile range of flux in Stokes I
     * Fraction of Stokes I pixels >= 0.5 x the peak
     * Median flux in Stokes |V|
     * Inter-quartile range of flux in Stokes |V|
     * Fraction of Stokes |V| pixels >= 0.5 x the peak
    """
    
    reshape_needed = False
    if len(sky_i.shape) == 1:
        reshape_needed = True
        sky_i = sky_i.reshape(1, *sky_i.shape)
        sky_v = sky_v.reshape(1, *sky_v.shape)
    nchan, npix = sky_i.shape
        
    imed = np.median(sky_i, axis=1)
    vmed = np.median(sky_v, axis=1)
    
    irms = np.sqrt((sky_i**2).mean(axis=1))
    vrms = np.sqrt((sky_v**2).mean(axis=1))

    iiqr = np.percentile(sky_i,75, axis=1) - np.percentile(sky_i,25, axis=1)
    viqr = np.percentile(sky_v,75, axis=1) - np.percentile(sky_v,25, axis=1)
    
    results = []
    for c in range(nchan):
        ihigh = len(np.where(sky_i[c] >= sky_i[c].max()*0.5)[0]) / npix
        vhigh = len(np.where(sky_v[c] >= sky_v[c].max()*0.5)[0]) / npix
        
        results.append({'med_i': imed[c],
                        'rms_i': irms[c],
                        'iqr_i': iiqr[c],
                        'frac_high_i': ihigh,
                        'med_v': vmed[c],
                        'rms_v': vrms[c],
                        'iqr_v': viqr[c],
                        'frac_high_v': vhigh
                       })
        
    if reshape_needed:
        results = results[0]
        
    return results


def extract_horizon(stokes_i: np.ndarray, stokes_v: np.ndarray, topo_wcs: WCS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the 1D numpy.ndarrays with the horizon values from the Stokes I
    and |V| images.
    """
    
    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    az, alt = _topo_wcs_to_altaz(xsize, ysize, topo_wcs)
    
    horizon = np.where(alt < HORIZON_ALT_DEG)
    hrz_az = az[horizon].ravel()
    order = np.argsort(hrz_az)
    
    hrz_i = np.zeros((nchan, len(horizon[0])), dtype=stokes_i.dtype)
    hrz_v = np.zeros((nchan, len(horizon[0])), dtype=stokes_v.dtype)
    for c in range(nchan):
        hrz_i[c] = stokes_i[c][horizon].ravel()[order]
        hrz_v[c] = stokes_v[c][horizon].ravel()[order]
        
    if reshape_needed:
        hrz_i = hrz_i[0]
        hrz_v = hrz_v[0]
        
    return hrz_i, hrz_v


def characterize_horizon(hrz_i: np.ndarray, hrz_v: np.ndarray) -> Union[Dict[str, float],
                                                                        List[Dict[str, float]]]:
    """
    Given the extracted Stokes I and |V| horizon, compute a few metrics
    to characterize it.  They are:
     * Peak flux in Stokes I
     * RMS flux in Stokes I
     * Inverse N-sigma level of the peak flux in Stokes I
     * Fraction of Stokes I pixels >= 0.5 x the peak
     * Peak flux in Stokes |V|
     * RMS flux in Stokes |V|
     * Inverse N-sigma level of the peak flux in Stokes |V|
     * Fraction of Stokes |V| pixels >= 0.5 x the peak
     * Stokes I to Stokes |V| high fraction ratio
    """
    
    reshape_needed = False
    if len(hrz_i.shape) == 1:
        reshape_needed = True
        hrz_i = hrz_i.reshape(1, *hrz_i.shape)
        hrz_v = hrz_v.reshape(1, *hrz_v.shape)
    nchan, npix = hrz_i.shape
    
    imax = hrz_i.max(axis=1)
    vmax = hrz_v.max(axis=1)
    
    irms = np.sqrt((hrz_i**2).mean(axis=1))
    vrms = np.sqrt((hrz_v**2).mean(axis=1))
    rrms = np.minimum(irms, vrms) / (np.maximum(irms, vrms) + 1e-8)
    rrms = np.clip(rrms, 0, 1)
    
    isig = hrz_i.std(axis=1) / (imax - np.median(hrz_i, axis=1) + 1e-8)
    vsig = hrz_v.std(axis=1) / (vmax - np.median(hrz_v, axis=1) + 1e-8)
    
    results = []
    for c in range(nchan):
        ihigh = len(np.where(hrz_i[c] >= imax[c]*0.5)[0]) / npix
        vhigh = len(np.where(hrz_v[c] >= vmax[c]*0.5)[0]) / npix
        rhigh = min(ihigh, vhigh) / (max(ihigh, vhigh) + 1e-8)
        rhigh = np.clip(rhigh, 0, 1)
        
        results.append({'max_i': imax[c],
                        'rms_i': irms[c],
                        'inv_sig_i': isig[c],
                        'frac_high_i': ihigh,
                        'max_v': vmax[c],
                        'rms_v': vrms[c],
                        'inv_sig_v': vsig[c],
                        'frac_high_v': vhigh,
                        'rms_ratio': rrms[c],
                        'frac_high_ratio': rhigh
                       })
        
    if reshape_needed:
        results = results[0]
        
    return results


def extract_1d_horizon(stokes_i: np.ndarray, stokes_v: np.ndarray, topo_wcs: WCS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the 1D numpy.ndarrays with the horizon values averaged over 0 to 30
    degrees elevation from the Stokes I and |V| images.
    """
    
    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    az, alt = _topo_wcs_to_altaz(xsize, ysize, topo_wcs)
    hrz_mask = alt < HORIZON_ALT_DEG
    
    hrz_i = np.zeros((nchan, 72), dtype=stokes_i.dtype)
    hrz_v = np.zeros((nchan, 72), dtype=stokes_v.dtype)
    for i in range(0, 360, 5):
        hbin = np.where(hrz_mask & (az >= i) & (az < i+5))
        for c in range(nchan):
            hrz_i[c,i//5] = np.mean(stokes_i[c][hbin])
            hrz_v[c,i//5] = np.mean(stokes_v[c][hbin])
            
    if reshape_needed:
        hrz_i = hrz_i[0]
        hrz_v = hrz_v[0]
        
    return hrz_i, hrz_v


def extract_beyond_horizon(stokes_i: np.ndarray, stokes_v: np.ndarray, topo_wcs: WCS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the 1D numpy.ndarrays with the values from beyond the horizon from
    the Stokes I and |V| images.
    """
    
    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    az, alt = _topo_wcs_to_altaz(xsize, ysize, topo_wcs)
    
    beyond = np.where(~np.isfinite(alt))
    
    byd_i = np.zeros((nchan, len(beyond[0])), dtype=stokes_i.dtype)
    byd_v = np.zeros((nchan, len(beyond[0])), dtype=stokes_v.dtype)
    for c in range(nchan):
        byd_i[c] = stokes_i[c][beyond].ravel()
        byd_v[c] = stokes_v[c][beyond].ravel()
        
    if reshape_needed:
        byd_i = byd_i[0]
        byd_v = byd_v[0]
        
    return byd_i, byd_v


def characterize_beyond_horizon(byd_i: np.ndarray, byd_v: np.ndarray) -> Union[Dict[str, float],
                                                                               List[Dict[str, float]]]:
    return characterize_horizon(byd_i, byd_v)


def extract_sources(stokes_i: np.ndarray, stokes_v: np.ndarray, timestamp: Time, wcs: WCS,
                    location: Optional[EarthLocation]=None,
                    srcs: List[str]=['CygA', 'CasA', 'TauA', 'VirA']) -> Union[Dict[str, List[np.ndarray]],
                                                                               List[Dict[str, np.ndarray]]]:
    """
    Return a set of postage stamps (both Stokes I and |V|) for the list of
    provided sources.
    """
    
    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    pc = wcs.pixel_to_world(*(wcs.wcs.crpix-1))
    srcs_xy = {}
    for src in srcs:
        try:
            sc = SOURCES[src]
        except KeyError:
            print(f"Unknown source '{src}', skipping")
            continue
            
        d = pc.separation(sc)
        if d.deg > HORIZON_ZA_DEG:
            continue
            
        y, x = wcs.world_to_pixel(sc)
        y, x = int(round(y.item())), int(round(x.item()))
        srcs_xy[src] = (y, x)
        
    results = []
    for c in range(nchan):
        regions = {}
        for src,(y,x) in srcs_xy.items():
            regions[src] = [stokes_i[c, x-7:x+8, y-7:y+8],
                            stokes_v[c, x-7:x+8, y-7:y+8]]
        results.append(regions)
        
    if reshape_needed:
        results = results[0]
        
    return results


def extract_sun(stokes_i: np.ndarray, stokes_v: np.ndarray, timestamp: Time, wcs: WCS,
                location: Optional[EarthLocation]=None) -> Union[Dict[str, List[np.ndarray]],
                                                                 List[Dict[str, np.ndarray]]]:
    """
    Similar to extract_sources but only works on the Sun.
    """

    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    pc = wcs.pixel_to_world(*(wcs.wcs.crpix-1))
        
    sc = get_body('sun', timestamp, location=location)
    sc = SkyCoord(sc.ra, sc.dec, frame='icrs')      # Otherwise we have problems with WCS
    d = pc.separation(sc)
    sun_y, sun_x = wcs.world_to_pixel(sc)
    
    results = []
    for c in range(nchan):
        regions = {}
        if d.deg <= HORIZON_ZA_DEG:
            y, x = int(round(sun_y.item())), int(round(sun_x.item()))
            regions['sun'] = [stokes_i[c, x-7:x+8, y-7:y+8],
                              stokes_v[c, x-7:x+8, y-7:y+8]]
            if regions['sun'][0].size == 0:
                del regions['sun']
        results.append(regions)
        
    if reshape_needed:
        results = results[0]
        
    return results


def extract_jupiter(stokes_i: np.ndarray, stokes_v: np.ndarray, timestamp: Time, wcs: WCS,
                    location: Optional[EarthLocation]=None) -> Union[Dict[str, List[np.ndarray]],
                                                                     List[Dict[str, np.ndarray]]]:
    """
    Similar to extract_sources but only works on Jupiter.
    """
    
    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    pc = wcs.pixel_to_world(*(wcs.wcs.crpix-1))
    
    sc = get_body('jupiter', timestamp, location=location)
    sc = SkyCoord(sc.ra, sc.dec, frame='icrs')      # Otherwise we have problems with WCS
    d = pc.separation(sc)
    jupiter_x, jupiter_y = wcs.world_to_pixel(sc)
   
    results = []
    for c in range(nchan):
        regions = {}
        if d.deg <= HORIZON_ZA_DEG: 
            y, x = int(round(jupiter_y.item())), int(round(jupiter_x.item()))
            regions['jupiter'] = [stokes_i[c, x-7:x+8, y-7:y+8],
                                  stokes_v[c, x-7:x+8, y-7:y+8]]
            if regions['jupiter'][0].size == 0:
                del regions['jupiter']
        results.append(regions)
        
    if reshape_needed:
        results = results[0]
            
    return results


def characterize_sources(regions: Union[Dict[str, float], List[Dict[str, float]]]) -> Union[Dict[str, float],
                                                                                            List[Dict[str, float]]]:
    """
    Given a dictionary of postage stamps, characterize the sources and return
    a set of metrics.  The are:
     * Number of sources analyzed
     * The peak background subtracted flux measured
     * Average ratio of the background to the peak flux
     * Average ratio of the X to Y FWHM estimates
     * Average ratio of Stokes |V| to I at the Stokes I peak location
    """
    
    reshape_needed = False
    if not isinstance(regions, list):
        reshape_needed = True
        regions = [regions,]
        
    results = []
    for c_region in regions:
        c_results = {}
        for src,(stokes_i,stokes_v) in c_region.items():
            tot = stokes_i.sum() + 1e-8
            xp = stokes_i.sum(axis=0)
            yp = stokes_i.sum(axis=1)
            i = np.arange(xp.size)
            j = np.arange(yp.size)
            
            cx = (xp*i).sum() / tot
            cy = (yp*j).sum() / tot
            
            wx = np.sqrt((xp*(i-cx)**2).sum() / tot)
            wy = np.sqrt((yp*(j-cy)**2).sum() / tot)
            
            bk = np.median(stokes_i)
            pk = stokes_i.max()
            
            c = np.where(stokes_i == pk)
            pf = stokes_v[c][0] / (stokes_i[c][0] + 1e-8)
            
            c_results[src] = {'flux': pk - bk,
                              'background': bk,
                              'center_x': cx,
                              'center_y': cy,
                              'width_x': wx,
                              'width_y': wy,
                              'v_over_i': pf
                             }
            
        nsrc = 0
        peak_flux = 0.0
        mean_contrast = 0.0
        mean_width_ratio = 0.0
        mean_v_over_i = 0.0
        for src,res in c_results.items():
            peak_flux = max(peak_flux, res['flux'])
            mean_contrast += np.clip(res['background'] / (res['flux'] + 1e-8), 0, 1)
            mean_width_ratio += np.clip(min(res['width_x'], res['width_y']) / (max(res['width_x'], res['width_y']) + 1e-8), 0, 1)
            mean_v_over_i += np.clip(res['v_over_i'], 0, 1)
            nsrc += 1
            
        if nsrc > 0:
            mean_contrast /= nsrc
            mean_width_ratio /= nsrc
            mean_v_over_i /= nsrc
        else:
            mean_contrast = 1.0
            mean_width_ratio = 1.0
            mean_v_over_i = 0.0
            
        results.append({'nsource': nsrc,
                        'peak_flux': peak_flux,
                        'mean_contrast': mean_contrast,
                        'mean_width_ratio': mean_width_ratio,
                        'mean_v_over_i': mean_v_over_i
                       })
        
    if reshape_needed:
        results = results[0]
        
    return results

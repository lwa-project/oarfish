import numpy as np
from scipy.ndimage import gaussian_filter
from functools import lru_cache
from typing import Tuple, Union, List, Dict, Optional

from astropy.coordinates import SkyCoord, get_body, EarthLocation
from astropy.time import Time
from astropy.wcs import WCS
from astropy import units as u


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
def station_to_earthlocation(station_id: Union[str, bytes]) -> EarthLocation:
    """
    Given a station name return an EarthLocation that corresponds to it.
    """
    
    if isinstance(station_id, bytes):
        station_id = station_id.decode()
        
    station_id = station_id.replace('-', '').lower()
    if station_id == 'lwa1':
        # LWA1 coordinates
        station_location = EarthLocation.from_geodetic(
            lon=-107.628350 * u.deg,
            lat=34.068894 * u.deg,
            height=2133.6 * u.m
        )
        
    elif station_id == 'lwasv':
        # LWA-SV coordinates
        station_location = EarthLocation.from_geodetic(
            lon=-106.885783*u.deg,
            lat=34.348358*u.deg,
            height=1477.8*u.m
        )
        
    elif station_id == 'lwana':
        # LWA-NA coordinates
        station_location = EarthLocation.from_geodetic(
            lon=-107.640 * u.deg,
            lat=34.247 * u.deg,
            height=2133.6 * u.m
        )
        
    else:
        raise ValueError(f"Unknown station '{station_id}'")
        
    return station_location


def get_time_references(timestamp: Time, wcs: WCS,
                        location: Optional[EarthLocation]=None) -> Tuple[float, float]:
    """
    Given a timestamp, a WCS instance, and an optional EarthLocation, return
    a two-element tuple of:
     * the approximate local sidereal time in hours
     * the approximate local apparent solar time in hours (0 = midnight).
    """
    
    lst = wcs.wcs.crval[0] % 360 / 15.0
    sc = get_body('sun', timestamp, location=location)
    ha = (lst - sc.ra.hourangle) % 24
    lt = (12 + ha) % 24
    
    return lst, lt


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
                    location: Optional[EarthLocation]=None, window_size: int=15,
                    srcs: List[str]=['CygA', 'CasA', 'TauA', 'VirA']) -> Union[Dict[str, Dict],
                                                                               List[Dict[str, Dict]]]:
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
    
    if window_size % 2 == 0:
        window_size += 1
    wpad = window_size //2
    
    pc = wcs.pixel_to_world(*(wcs.wcs.crpix-1))
    srcs_xyz = {}
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
        srcs_xyz[src] = (y, x, d.deg)
        
    results = []
    for c in range(nchan):
        regions = {}
        for src,(y,x,z) in srcs_xyz.items():
            regions[src] = {'stokes_i': stokes_i[c, x-wpad:x+wpad+1, y-wpad:y+wpad+1],
                            'stokes_v': stokes_v[c, x-wpad:x+wpad+1, y-wpad:y+wpad+1],
                            'zenith_angle': z,
                            'timestamp': timestamp
                           }
        results.append(regions)
        
    if reshape_needed:
        results = results[0]
        
    return results


def extract_sun(stokes_i: np.ndarray, stokes_v: np.ndarray, timestamp: Time, wcs: WCS,
                location: Optional[EarthLocation]=None,
                window_size: int=15) -> Union[Dict[str, Dict],
                                              List[Dict[str, Dict]]]:
    """
    Similar to extract_sources but only works on the Sun.
    """

    reshape_needed = False
    if len(stokes_i.shape) == 2:
        reshape_needed = True
        stokes_i = stokes_i.reshape(1, *stokes_i.shape)
        stokes_v = stokes_v.reshape(1, *stokes_v.shape)
    nchan, xsize, ysize = stokes_i.shape
    
    if window_size % 2 == 0:
        window_size += 1
    wpad = window_size //2
    
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
            regions['sun'] = {'stokes_i': stokes_i[c, x-wpad:x+wpad+1, y-wpad:y+wpad+1],
                              'stokes_v': stokes_v[c, x-wpad:x+wpad+1, y-wpad:y+wpad+1],
                              'zenith_angle': d.deg,
                              'timestamp': timestamp
                             }
            if regions['sun']['stokes_i'].size == 0:
                del regions['sun']
        results.append(regions)
        
    if reshape_needed:
        results = results[0]
        
    return results


def extract_jupiter(stokes_i: np.ndarray, stokes_v: np.ndarray, timestamp: Time, wcs: WCS,
                    location: Optional[EarthLocation]=None,
                    window_size: int=15) -> Union[Dict[str, List[np.ndarray]],
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
    
    if window_size % 2 == 0:
        window_size += 1
    wpad = window_size //2
    
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
            regions['jupiter'] = {'stokes_i': stokes_i[c, x-wpad:x+wpad+1, y-wpad:y+wpad+1],
                                  'stokes_v': stokes_v[c, x-wpad:x+wpad+1, y-wpad:y+wpad+1],
                                  'zenith_angle': d.deg,
                                  'timestamp': timestamp
                                 }
            if regions['jupiter']['stokes_i'].size == 0:
                del regions['jupiter']
        results.append(regions)
        
    if reshape_needed:
        results = results[0]
            
    return results


@lru_cache(maxsize=64)
def get_baars_flux(source: str, frequency: float, timestamp: Optional[Time]=None) -> float:
    """
    Given a source name, an observing frequency in Hz, and an optional
    timestamp, return the Baars et al. (1977) flux density of source in Jy.
    """
    
    log_freq = np.log10(frequency/1e6)
    
    if source == 'CygA':
        log_flux = 4.695 + 0.085*log_freq - 0.178*log_freq**2
    elif source == 'CasA':
        log_flux = 5.625 - 0.634*log_freq - 0.023*log_freq**2
    elif source == 'TauA':
        log_flux = 3.915 - 0.299*log_freq
    elif source == 'VirA':
        log_flux = 5.023 - 0.856*log_freq
    else:
        raise ValueError(f"Unknown source '{source}'")
        
    flux = 10**log_flux
    
    # Apply the secular decay to Cas A
    if source == 'CasA':
        age = 2025 - 1965
        if timestamp is not None:
            age = timestamp.jyear - 1965
        flux *= (1 - 0.008)**age
        
    return flux


def get_approx_beam_corr(zenith_angle: float) -> float:
    """
    Use the beam response correction terms from Schinzel & Polisensky (2014,
    LWA Memo #202) to come up a rough primary beam correction to apply to
    observed source flux densities.
    """
    
    # Convert to altitude since the #202 equations are in that
    altitude = 90 - zenith_angle
    
    # Compute XX, YY, and then average
    xx = 155*altitude**-1.55 + 0.84
    yy = 14.1*altitude**-0.62 + 0.10
    
    return max(1, (xx + yy)/2.0)


def characterize_sources(regions: Union[Dict[str, Dict], List[Dict[str, Dict]]])-> Union[Dict[str, float],
                                                         List[Dict[str, float]]]:
    """
    Given a dictionary of postage stamps, characterize the sources and return
    a set of metrics.  The are:
     * Number of sources analyzed
     * The peak background subtracted flux measured
     * Average ratio of the background to the peak flux
     * Average ratio of the minor to major axes FWHM estimates
     * Average value of the position angle of the major axis
     * Average ratio of Stokes |V| to I at the Stokes I peak location
     * Average flux scale factor to get from primary beam corrected observed
       fluxes to Jy
    """
    
    reshape_needed = False
    if not isinstance(regions, list):
        reshape_needed = True
        regions = [regions,]
        
    results = []
    for c_region in regions:
        c_results = {}
        for src,data in c_region.items():
            stokes_i = data['stokes_i']
            stokes_v = data['stokes_v']
            zenith_angle = data['zenith_angle']
            timestamp = data['timestamp']
            
            tot = stokes_i.sum() + 1e-8
            j, i = np.indices(stokes_i.shape)
            
            cx = (stokes_i*i).sum() / tot
            cy = (stokes_i*j).sum() / tot
            
            wx2 = ((i - cx)**2 * stokes_i).sum() / tot
            wy2 = ((j - cy)**2 * stokes_i).sum() / tot
            wxy = ((i - cx)*(j - cy) * stokes_i).sum() / tot
            
            cov = np.array([[wx2, wxy], [wxy, wy2]]) + np.eye(2)*1e-8
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            wx = np.sqrt(evals[0])
            wy = np.sqrt(evals[1])
            th = np.arctan2(evecs[1,0], evecs[0,0]) % np.pi
            
            bk = np.median(np.stack([stokes_i[0,:],
                                     stokes_i[-1,:],
                                     stokes_i[:,0],
                                     stokes_i[:,-1]]))
            pk = stokes_i.max()
            
            c = np.where(stokes_i == pk)
            pf = stokes_v[c][0] / (stokes_i[c][0] + 1e-8)
            
            c_results[src] = {'flux': pk - bk,
                              'background': bk,
                              'center_x': cx,
                              'center_y': cy,
                              'width_maj': wx,
                              'width_min': wy,
                              'pos_ang': np.rad2deg(th),
                              'v_over_i': pf
                             }
            
        nsrc = 0
        peak_flux = 0.0
        mean_contrast = 0.0
        mean_width_ratio = 0.0
        mean_pa = 0.0
        mean_v_over_i = 0.0
        for src,res in c_results.items():
            peak_flux = max(peak_flux, res['flux'])
            mean_contrast += np.clip(res['background'] / (res['flux'] + 1e-8), 0, 1)
            mean_width_ratio += np.clip(min(res['width_maj'], res['width_min']) / (max(res['width_maj'], res['width_min']) + 1e-8), 0, 1)
            mean_pa += res['pos_ang']
            mean_v_over_i += np.clip(res['v_over_i'], 0, 1)
            nsrc += 1
            
        if nsrc > 0:
            mean_contrast /= nsrc
            mean_width_ratio /= nsrc
            mean_pa /= nsrc
            mean_v_over_i /= nsrc
        else:
            mean_contrast = 1.0
            mean_width_ratio = 1.0
            mean_pa = 0.0
            mean_v_over_i = 0.0
            
        results.append({'nsource': nsrc,
                        'peak_flux': peak_flux,
                        'mean_contrast': mean_contrast,
                        'mean_width_ratio': mean_width_ratio,
                        'mean_pos_ang': mean_pa,
                        'mean_v_over_i': mean_v_over_i
                       })
        
    if reshape_needed:
        results = results[0]
        
    return results

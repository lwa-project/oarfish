import numpy as np
from copy import deepcopy
from typing import Tuple, Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from astropy.time import Time
from astropy.coordinates import get_sun, get_body, AltAz, EarthLocation, SkyCoord, ICRS
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u

from .utils import *


def info_to_wcs(info: Dict[str,Any], image_size: Optional[int]=None) -> Tuple[WCS, WCS]:
    """
    Given a .oims metadata dictionary, construct RA/dec and azimuth/altitude
    WCS instances and return them.
    """
    
    if image_size is None:
        try:
            image_size = info['ngrid']
        except KeyError:
            raise RuntimeError("'image_size' not provided and not in the metadata")
            
    pixel_scale = info['pixel_size']
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(image_size + 1)/2.0,
                     (image_size + 1)/2.0]
    wcs.wcs.cdelt = np.array([-pixel_scale,
                              pixel_scale])
    wcs.wcs.crval = [info['center_ra'],
                     info['center_dec']]
    wcs.wcs.ctype = ["RA---SIN",
                     "DEC--SIN"]
    
    # Topocenetric WCS
    center_alt = center_az = 90.0
    if 'center_alt' in info and 'center_az' in info:
        center_alt = info['center_alt']
        center_az = info['center_az']
    
    topo_wcs =  WCS(naxis=2)
    topo_wcs.wcs.crpix = [(image_size + 1)/2.0,
                          (image_size + 1)/2.0]
    topo_wcs.wcs.cdelt = np.array([-pixel_scale,
                                   pixel_scale])
    topo_wcs.wcs.crval = [center_az,
                          center_alt]
    topo_wcs.wcs.ctype = ["RA---SIN",    # Really azimuth
                          "DEC--SIN"]    # Really altitude

    # Off-zenith phase center correction
    if center_alt != 90.0:
        zen_c = (90 - center_alt) * np.pi/180
        zaz_c = (center_az + 180) * np.pi/180
        
        xi = np.sin(zen_c) * np.cos(zaz_c)
        eta = np.sin(zen_c) * np.sin(zaz_c)
        
        wcs.wcs.set_pv([(2,1,xi), (2,2,eta)])
        topo_wcs.wcs.set_pv([(2,1,xi), (2,2,eta)])
        
    return wcs, topo_wcs


def load_lwatv_data(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict[str,Any]]:
    """
    Given a .npz snapshot from a LWATV .oims file, load in Stokes I and |V|
    and return them along with the frame metadata.
    """
    
    data = np.load(filename, allow_pickle=True)
    si = data['data'][0,0,:,:]
    sv = np.abs(data['data'][0,-1,:,:])
    si[np.where(~np.isfinite(si))] = 0.0
    sv[np.where(~np.isfinite(sv))] = 0.0
    info = data['info'].item()
    data.close()
    
    return si,sv,info


class LWATVDataset(Dataset):
    def __init__(self, image_paths: Union[str, List[str]], labels: Optional[int]=None,
                       transform: Optional[Any]=None, station_location: Optional[EarthLocation]=None):
        if not isinstance(image_paths, (tuple, list)):
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.labels = labels
        if transform is None:
            self.transform = transforms.Compose([
            transforms.RandomRotation(15),  # Small rotations since orientation matters
            transforms.RandomAffine(0, translate=(0.05, 0.05)),  # Small translations
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight intensity variations
            *self.default_transform().transforms
        ])
        
        # Default to LWA-SV location if none provided
        if station_location is None:
            station_location = EarthLocation(lat=34.348358*u.deg, 
                                             lon=-106.885783*u.deg, 
                                             height=1477.8*u.m
                                            )
        self.location = station_location
        
    @staticmethod
    def default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to handle variable sizes
            # No need for ToTensor() since we're already converting numpy to tensor
            # Normalization is handled in __getitem__ if not specified here
        ])
    
    @staticmethod
    def _process_image_pair(metadata: Dict[str, Any], stokes_i: np.ndarray, stokes_v: np.ndarray,
                            location: Optional[EarthLocation]=None,
                            transform: Optional[Any]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build up the WCS
        wcs, topo_wcs = info_to_wcs(metadata, image_size=stokes_i.shape[0])
        
        # Normalization
        vmin, vmax = 0, max([np.percentile(stokes_i.ravel(), 99.75), 1e-8])
        stokes_i = np.clip(stokes_i, vmin, vmax) / vmax
        stokes_v = np.clip(stokes_v, vmin, vmax) / vmax
        
         # Get astronomical features
        timestamp = Time(metadata['start_time'], format='mjd', scale='utc')
        lst = wcs.wcs.crval[0] % 360
       
        sky_i, sky_v = extract_sky(stokes_i, stokes_v, topo_wcs) 
        hrz_i, hrz_v = extract_horizon(stokes_i, stokes_v, wcs)
        byd_i, byd_v = extract_beyond_horizon(stokes_i, stokes_v, topo_wcs)
        srcs = extract_sources(stokes_i, stokes_v, timestamp, wcs,
                               location=location, window_size=15)
        sun = extract_sun(stokes_i, stokes_v, timestamp, wcs,
                          location=location, window_size=15)
        jupiter = {}
        if metadata['start_freq'] <= 40e6:
            jupiter = extract_jupiter(stokes_i, stokes_v, timestamp, wcs,
                                      location=location, window_size=15)
        
        # Analyze astronoical features
        sky = characterize_sky(sky_i, sky_v)
        hrz = characterize_horizon(hrz_i, hrz_v)
        byd = characterize_beyond_horizon(byd_i, byd_v)
        srcs = characterize_sources(srcs)
        sun = characterize_sources(sun)
        jupiter = characterize_sources(jupiter)
        
        # Extract the horizon and apply a small +/- 15 degree rotation manually
        hi, hv = extract_1d_horizon(stokes_i, stokes_v, topo_wcs)
        rr = int(round(np.random.rand()*6-3))
        hi = np.roll(hi, rr)
        hv = np.roll(hv, rr)
        
        # Convert to tensors
        stokes_i = torch.from_numpy(stokes_i).float().unsqueeze(0)
        stokes_v = torch.from_numpy(stokes_v).float().unsqueeze(0)
        hi = torch.from_numpy(hi).float()
        hv = torch.from_numpy(hv).float()
        
        # Apply transforms if any
        if transform:
            stokes_i = transform(stokes_i)
            stokes_v = transform(stokes_v)
        
        # Combine channels
        img_tensor = torch.cat([stokes_i, stokes_v], dim=0)
        hrz_tensor = torch.stack([hi, hv], dim=0)
        
        # Get astronomical features
        astro_tensor = torch.tensor([
            sun['mean_contrast'],
            sun['mean_v_over_i'],
            jupiter['mean_contrast'],
            jupiter['mean_v_over_i'],
            srcs['mean_contrast'],
            srcs['mean_width_ratio'],
            srcs['mean_v_over_i'],
            byd['rms_i'],
            byd['rms_v'],
            hrz['max_i'],
            hrz['rms_i'],
            hrz['frac_high_i'],
            hrz['max_v'],
            hrz['rms_v'],
            hrz['frac_high_v'],
            hrz['frac_high_ratio'],
            sky['med_i'],
            sky['med_v'],
            sky['iqr_i'],
            sky['iqr_v'],
            lst/360,
            metadata['start_freq']/98e6
        ], dtype=torch.float32)
        
        return img_tensor, hrz_tensor, astro_tensor
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        img_path = self.image_paths[idx]
        
        # Load data
        try:
            si, sv, metadata = load_lwatv_data(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return a different sample
            return self.__getitem__((idx + 1) % len(self))
            
        # Process
        img_tensor, hrz_tensor, astro_tensor = self._process_image_pair(metadata,
                                                                        si, sv,
                                                                        self.location,
                                                                        self.transform)
        # Done
        if self.labels is not None:
            label = self.labels[idx]
            return img_tensor, hrz_tensor, astro_tensor, label
        return img_tensor, hrz_tensor, astro_tensor


class SingleChannelDataset(LWATVDataset):
    def __init__(self, metadata: Dict[str, Any], stokes_i: np.ndarray, stokes_v: np.ndarray,
                       labels: Optional[int]=None, transform: Optional[Any]=None,
                       station_location: Optional[EarthLocation]=None):
        super().__init__([''], labels=labels, transform=transform,
                         station_location=station_location)
        if len(stokes_i.shape) != 2 or len(stokes_v.shape) != 2:
            raise RuntimeError("Expected a single frequency image for Stokes I and |V|")
        self._metadata = metadata
        self._stokes_i = stokes_i
        self._stokes_v = stokes_v
        
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        if idx != 0:
            raise RuntimeError("SingleChannelDataset only contains a single integration/channel")
            
        img_tensor, hrz_tensor, astro_tensor = self._process_image_pair(self._metadata,
                                                                        self._stokes_i,
                                                                        self._stokes_v,
                                                                        self.location,
                                                                        self.transform)
        
        if self.labels is not None:
            label = self.labels[idx]
            return img_tensor, hrz_tensor, astro_tensor, label
        return img_tensor, hrz_tensor, astro_tensor


class MultiChannelDataset(LWATVDataset):
    def __init__(self, metadata: Dict[str, Any], stokes_i: np.ndarray, stokes_v: np.ndarray,
                       labels: Optional[int]=None, transform: Optional[Any]=None,
                       station_location: Optional[EarthLocation]=None):
        nchan = stokes_i.shape[0]
        super().__init__(['']*nchan, labels=labels, transform=transform,
                         station_location=station_location)
        if len(stokes_i.shape) != 3 or len(stokes_v.shape) != 3:
            raise RuntimeError("Expected image cube for Stokes I and |V|")
        self.nchan = nchan
        self._metadata = metadata
        self._stokes_i = stokes_i
        self._stokes_v = stokes_v
        
        self._cache = {}
        
    @staticmethod
    def _process_image_stack(metadata: Dict[str, Any], stokes_i: np.ndarray, stokes_v: np.ndarray,
                             location: Optional[EarthLocation]=None,
                             transform: Optional[Any]=None) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(stokes_i.shape) == 2:
            stokes_i = stokes_i.reshape(1, *stokes_i.shape)
            stokes_v = stokes_v.reshape(1, *stokes_v.shape)
        nchan, xsize, ysize = stokes_i.shape
        
        # Build up the WCS
        wcs, topo_wcs = info_to_wcs(metadata, image_size=xsize)
        
        # Normalization
        for c in range(nchan):
            vmin, vmax = 0, max([np.percentile(stokes_i[c].ravel(), 99.75), 1e-8])
            stokes_i[c] = np.clip(stokes_i[c], vmin, vmax) / vmax
            stokes_v[c] = np.clip(stokes_v[c], vmin, vmax) / vmax
        
         # Get astronomical features
        timestamp = Time(metadata['start_time'], format='mjd', scale='utc')
        lst = wcs.wcs.crval[0] % 360
       
        sky_i, sky_v = extract_sky(stokes_i, stokes_v, topo_wcs) 
        hrz_i, hrz_v = extract_horizon(stokes_i, stokes_v, wcs)
        byd_i, byd_v = extract_beyond_horizon(stokes_i, stokes_v, topo_wcs)
        srcs = extract_sources(stokes_i, stokes_v, timestamp, wcs,
                               location=location, window_size=15)
        sun = extract_sun(stokes_i, stokes_v, timestamp, wcs,
                          location=location, window_size=15)
        jupiter = extract_jupiter(stokes_i, stokes_v, timestamp, wcs,
                                  location=location, window_size=15)
        for c in range(nchan):
            f = metadata['start_freq'] + c*metadata['bandwidth']
            if f > 40e6:
                jupiter[c] = {}
                
        # Analyze astronoical features
        sky = characterize_sky(sky_i, sky_v)
        hrz = characterize_horizon(hrz_i, hrz_v)
        byd = characterize_beyond_horizon(byd_i, byd_v)
        srcs = characterize_sources(srcs)
        sun = characterize_sources(sun)
        jupiter = characterize_sources(jupiter)
        
        # Extract the horizon and apply a small +/- 15 degree rotation manually
        hi, hv = extract_1d_horizon(stokes_i, stokes_v, topo_wcs)
        rr = int(round(np.random.rand()*6-3))
        hi = np.roll(hi, rr, axis=1)
        hv = np.roll(hv, rr, axis=1)
        
        # Convert to output tuples
        finals = []
        for c in range(nchan):
            ## Convert to tensors
            c_stokes_i = torch.from_numpy(stokes_i[c]).float().unsqueeze(0)
            c_stokes_v = torch.from_numpy(stokes_v[c]).float().unsqueeze(0)
            c_hi = torch.from_numpy(hi[c]).float()
            c_hv = torch.from_numpy(hv[c]).float()
            
            ## Apply transforms if any
            if transform:
                c_stokes_i = transform(c_stokes_i)
                c_stokes_v = transform(c_stokes_v)
            
            ## Combine channels
            img_tensor = torch.cat([c_stokes_i, c_stokes_v], dim=0)
            hrz_tensor = torch.stack([c_hi, c_hv], dim=0)
            
            ## Get astronomical features
            astro_tensor = torch.tensor([
                sun[c]['mean_contrast'],
                sun[c]['mean_v_over_i'],
                jupiter[c]['mean_contrast'],
                jupiter[c]['mean_v_over_i'],
                srcs[c]['mean_contrast'],
                srcs[c]['mean_width_ratio'],
                srcs[c]['mean_v_over_i'],
                byd[c]['rms_i'],
                byd[c]['rms_v'],
                hrz[c]['max_i'],
                hrz[c]['rms_i'],
                hrz[c]['frac_high_i'],
                hrz[c]['max_v'],
                hrz[c]['rms_v'],
                hrz[c]['frac_high_v'],
                hrz[c]['frac_high_ratio'],
                sky[c]['med_i'],
                sky[c]['med_v'],
                sky[c]['iqr_i'],
                sky[c]['iqr_v'],
                lst/360,
                (metadata['start_freq'] + c*metadata['bandwidth'])/98e6
            ], dtype=torch.float32)
            
            finals.append((img_tensor, hrz_tensor, astro_tensor))
        
        return finals
        
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        if idx > self._stokes_i.shape[0]:
            raise RuntimeError("Requested too many channels for this data set")
            
        if idx in self._cache:
            return self._cache[idx]
            
        # Build full list of tensors and features
        finals = self._process_image_stack(self._metadata,
                                           self._stokes_i, self._stokes_v,
                                           self.location, self.transform)
        
        for chan in range(self.nchan):
            ## Copy the header and make an update for the current frequency
            chan_info = deepcopy(self._metadata)
            chan_info['start_freq'] += chan*chan_info['bandwidth']
            chan_info['stop_freq'] = chan_info['start_freq']
            
            ## Pull out the channel we want
            img_tensor, hrz_tensor, astro_tensor = finals[chan]
            
            self._cache[chan] = (img_tensor, hrz_tensor, astro_tensor, chan_info)
            
        return self._cache[idx]

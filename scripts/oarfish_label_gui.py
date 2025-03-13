import os
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
from copy import deepcopy
from PIL import Image, ImageTk, ImageDraw
import logging
from datetime import datetime
from functools import lru_cache
import argparse

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, get_body, SkyCoord, AltAz
from astropy.wcs import WCS

from oarfish.data import info_to_wcs
from oarfish.utils import station_to_earthlocation, SOURCES, EXT_SOURCES

try:
    from lsl_toolkits.OrvilleImage import OrvilleImageDB
except ImportError:
    from OrvilleImageDB import OrvilleImageDB
    
ALLOW_PIMS = False
try:
    from lsl_toolkits.PasiImage import PasiImageDB
    ALLOW_PIMS = True
except ImportError:
    try:
        from PasiImageDB import PasiImageDB
        ALLOW_PIMS = True
    except ImportError:
        pass


class FocusAreaMismatch(RuntimeError):
    pass


class ImageClassifierApp:
    def __init__(self, root, data_dir: str, output_dir: str, colormap='viridis',
                       is_multi_class: bool=True,
                       focus_sun: bool=False, focus_jupiter: bool=False):
        self.root = root
        self.root.title("LWATV Image Classifier")
        
        # Configure paths
        self.base_path = os.path.normpath(data_dir)
        self.output_base = os.path.normpath(output_dir)
        
        # Colormap to use
        self.cmap = colormap
        
        # Focus areas
        self.focus_sun = focus_sun
        self.focus_jupiter = focus_jupiter
        
        # Define classification categories
        if is_multi_class:
            self.categories = {
                'g': 'good',
                'm': 'medium_rfi',
                'r': 'high_rfi',
                's': 'sun',
                'j': 'jupiter',
                'c': 'corrupted',
            }
            
            self.buttons = [
                ("Good (g)", 'good'),
                ("Medium RFI (m)", 'medium_rfi'),
                ("High RFI (r)", 'high_rfi'),
                ("Sun (s)", 'sun'),
                ("Jupiter (j)", 'jupiter'),
                ("Corrupted (c)", 'corrupted'),
                ("Skip (k)", None)
            ]
            
        else:
            self.categories = {
                'g': 'good',
                'b': 'bad',
            }
            
            self.buttons = [
                ("Good (g)", 'good'),
                ("Bad (b)", 'bad'),
                ("Skip (k)", None)
            ]
            
        self.setup_directories()
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.output_base, 'classification_log.txt'),
            level=logging.INFO,
            format='%(asctime)s,%(message)s'
        )
        
        # Initialize state
        self.current_file = None
        self.current_image_info = None
        self.stats = {cat: 0 for cat in self.categories.values()}
        self.stats['total'] = 0
        
        # Create GUI elements
        self.setup_gui()
        
        # Bind keyboard events
        for key, category in self.categories.items():
            self.root.bind(key, lambda e, cat=category: self.classify_image(cat))
        self.root.bind('k', lambda e: self.load_random_image())  # 'k' for skip
        
        # Load first image
        self.load_random_image()

    def setup_directories(self):
        """Create output directories if they don't exist"""
        for category in self.categories.values():
            os.makedirs(os.path.join(self.output_base, category), exist_ok=True)

    def setup_gui(self):
        """Setup the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Info label
        self.info_label = ttk.Label(main_frame, text="")
        self.info_label.grid(row=1, column=0, columnspan=4)
        
        # Stats label
        self.stats_label = ttk.Label(main_frame, text=self.get_stats_text())
        self.stats_label.grid(row=2, column=0, columnspan=4, pady=5)
        
        # Control buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=5)
        
        # Add buttons for each category
        for i, (text, category) in enumerate(self.buttons):
            cmd = self.load_random_image if category is None else lambda cat=category: self.classify_image(cat)
            ttk.Button(btn_frame, text=text, command=cmd).grid(row=i//4, column=i%4, padx=5, pady=2)

    def get_stats_text(self):
        """Generate statistics text"""
        stats_list = [f"{cat.title()}: {count}" for cat, count in self.stats.items()]
        return " | ".join(stats_list)

    def create_matplotlib_figure(self, data, wcs, channel=0):
        """Create a matplotlib figure with the image and celestial object markers"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display the image with grayscale colormap
        image_data = data[channel, 0, ...]
        vmin, vmax = 0.0, np.percentile(image_data, 99.75)
        ax.imshow(image_data, cmap=self.cmap, vmin=vmin, vmax=vmax, origin='lower')
        
        # Get timestamp from current image info
        timestamp = Time(self.current_image_info['info']['start_time'], format='mjd')
        
        # Add frequency to the plot
        freq = self.current_image_info['info']['start_freq'] + channel * self.current_image_info['info']['bandwidth']
        ax.text(0.02, 0.98, f"{freq/1e6:.1f} MHz", 
                transform=ax.transAxes, color='white', 
                fontsize=12, verticalalignment='top')
        if 'station' in self.current_image_info['info']:
            station_id = self.current_image_info['info']['station']
            if 'visFileName' in self.current_image_info['info']:
                station_id += '\n(PASI)'
            ax.text(0.90, 0.98, station_id, 
                    transform=ax.transAxes, color='white', 
                    fontsize=12, verticalalignment='top')
        
        if wcs is not None:
            ## Try to get a location for the station
            try:
                el = station_to_earthlocation(self.current_image_info['info']['station'])
                altaz_frame = AltAz(obstime=timestamp, location=el)
            except ValueError:
                altaz_frame = None
                
            ## Plot the major A team sources
            for src,sc in SOURCES.items():
                try:
                    sc_x, sc_y = wcs.world_to_pixel(sc)
                    if np.isfinite(sc_x) and np.isfinite(sc_y):
                        ax.text(sc_x+2, sc_y+2, src, color='white', fontsize=12)
                except Exception as e:
                    pass
                
            ## Plot the minor A team sources
            for src,sc in EXT_SOURCES.items():
                try:
                    sc_x, sc_y = wcs.world_to_pixel(sc)
                    if np.isfinite(sc_x) and np.isfinite(sc_y):
                        ax.text(sc_x+2, sc_y+2, src, color='white', fontsize=9)
                except Exception as e:
                    pass
                    
            ## Plot the Sun and jupiter
            for body in ('sun', 'jupiter'):
                bdy = get_body(body, timestamp)
                bdy = SkyCoord(bdy.ra, bdy.dec, frame='icrs')
                if altaz_frame is not None:
                    bdy_alt = bdy.transform_to(altaz_frame)
                    bdy_alt = bdy_alt.alt.deg
                else:
                    bdy_alt = 90.0
                    
                try:
                    body = body.capitalize()
                    if bdy_alt > 30:
                        bdy_x, bdy_y = wcs.world_to_pixel(bdy)
                        if np.isfinite(bdy_x) and np.isfinite(bdy_y):
                            ax.text(bdy_x+2, bdy_y+2, body, color='white', fontsize=10)
                except Exception as e:
                    print(f"Error plotting position of {body}: {str(e)}")
                    
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    def display_image(self, data, channel=0):
        """Create matplotlib figure and display it in tkinter"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io

        # Create matplotlib figure
        fig = self.create_matplotlib_figure(data,
                                            self.current_image_info.get('wcs'),
                                            channel=channel)

        # Convert matplotlib figure to PNG
        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        plt.close(fig)  # Close the figure to free memory

        # Create PIL Image from PNG
        image = Image.open(buf)

        # Resize to maintain aspect ratio within 500x500
        max_size = 500
        ratio = min(max_size/image.width, max_size/image.height)
        new_size = (int(image.width*ratio), int(image.height*ratio))
        try:
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:
            image = image.resize(new_size, Image.ANTIALIAS)
            
        # Convert to PhotoImage for tkinter
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference!

        # Clean up
        buf.close()

    @staticmethod
    @lru_cache(maxsize=32)
    def _load_image_paths(base_path):
        image_paths = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.oims'):
                    image_paths.append(os.path.join(root, file))
                if ALLOW_PIMS and file.endswith('.pims'):
                    image_paths.append(os.path.join(root, file))
        return image_paths
        
    def load_random_image(self):
        """Load a random image from the database"""
        
        attempt = 0
        while True:
            attempt += 1
            if attempt % 100 == 0:
                print(f"Still looking after {attempt} attempts...")
                
            db = None
            try:
                # Get all image paths
                image_paths = self._load_image_paths(self.base_path)
                
                # Select random image
                filename = random.choice(image_paths)
                
                # Open the database
                db_path = filename
                if filename.endswith('.oims'):
                    db = OrvilleImageDB(db_path, 'r')
                else:
                    db = PasiImageDB(db_path, 'r')
                    
                # Get random image
                nint = len(db)
                image_index = random.randint(0, nint - 1)
                
                # Read the image
                hdr_data = db[image_index]
                info, data = hdr_data[0], hdr_data[1]
                if len(data.shape) == 3:
                    ## Must be PASI data.  Fix it up
                    ### Header
                    info = info.as_dict()
                    tstart = Time(info['startTime'], format='mjd', scale='tai')
                    tcen = Time(info['centroidTime'], format='mjd', scale='tai')
                    info['start_time'] = float(tstart.utc.mjd)
                    info['centroid_time'] = float(tcen.utc.mjd)
                    info['int_len'] = info['intLen']
                    info['start_freq'] = info['freq']
                    info['stop_freq'] = info['freq']
                    info['pixel_size'] = db.header.xPixelSize
                    info['stokes_params'] = db.header.stokesParams
                    if isinstance(info['stokes_params'], bytes):
                        info['stokes_params'] = info['stokes_params'].decode()
                    info['center_ra'] = info['zenithRA']
                    info['center_dec'] = info['zenithDec']
                    del info['worldreplace0']
                    
                    ### Data
                    data.shape = (1,)+data.shape
                    data = data.transpose(0,1,3,2)
                    
                if isinstance(data, np.ma.MaskedArray):
                    data = data.data
                try:
                    info['station'] = db.header.station
                    if isinstance(info['station'], bytes):
                        info['station'] = info['station'].decode()
                except KeyError:
                    pass
                wcs, _ = info_to_wcs(info, data.shape[-1])
                
                # Select a random channel
                channel_index = random.randint(0, data.shape[0]-1)
                
                # Apply focus area cuts
                if (self.focus_sun or self.focus_jupiter) and 'station' in info:
                    timestamp = Time(info['start_time'], format='mjd', scale='utc')
                    el = station_to_earthlocation(info['station'])
                    if self.focus_sun:
                        bdy = get_body('sun', timestamp)
                        bdy = bdy.transform_to(AltAz(obstime=timestamp, location=el))
                        if bdy.alt.deg < 30:
                            raise FocusAreaMismatch(f"Sun is at {bdy.alt.deg} deg altitude")
                        
                    elif self.focus_jupiter:
                        bdy = get_body('jupiter', timestamp)
                        bdy = bdy.transform_to(AltAz(obstime=timestamp, location=el))
                        if bdy.alt.deg < 30:
                            raise FocusAreaMismatch(f"Jupiter is at {bdy.alt.deg} deg altitude")
                        freq = info['start_freq'] + channel_index*info['bandwidth']
                        if freq < 10e6 or freq > 40e6:
                            raise FocusAreaMismatch(f"Frequency out of range for Jupiter at {freq/1e6} MHz")
                            
                # Store current image info
                self.current_image_info = {
                    'file': os.path.basename(filename),
                    'index': image_index,
                    'channel': channel_index,
                    'data': data,
                    'info': info,
                    'wcs': wcs  # Store WCS information for celestial object marking
                }
                
                self.display_image(data, channel=channel_index)
                    
                # Update info label
                self.info_label.config(
                    text=f"Source: {filename.replace(self.base_path+os.path.sep, '')} (Image {image_index}, Channel {channel_index})"
                )
                break
                
            except FocusAreaMismatch:
                pass
                
            except Exception as e:
                import traceback
                logging.error(f"Error loading image: {str(e)}")
                logging.error(f"Trackback: {traceback.format_exc()}")
                self.info_label.config(text=f"Error loading image: {str(e)}")
                
            finally:
                if db is not None:
                    db.close()
                    
    def classify_image(self, category):
        """Save the classification and load next image"""
        if not self.current_image_info:
            return
            
        try:
            # Save the raw data as .npz
            output_file = os.path.join(
                self.output_base,
                category,
                f"{self.current_image_info['file']}_{self.current_image_info['index']}_{self.current_image_info['channel']}.npz"
            )
            channel_index = self.current_image_info['channel']
            info = deepcopy(self.current_image_info['info'])
            info['start_freq'] += info['bandwidth']*channel_index
            np.savez(
                output_file,
                data=self.current_image_info['data'][[channel_index],...],
                info=info
            )
            
            # Log the classification
            log_msg = f"{self.current_image_info['file']},{self.current_image_info['index']},{self.current_image_info['channel']},{category}"
            logging.info(log_msg)
            
            # Update stats
            self.stats[category] += 1
            self.stats['total'] += 1
            self.stats_label.config(text=self.get_stats_text())
            
            # Load next image
            self.load_random_image()
            
        except Exception as e:
            logging.error(f"Error saving classification: {str(e)}")
            self.info_label.config(text=f"Error saving classification: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="tool to look at integrations from .oims or .pims files and label them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, default='.',
                        help='directory containing .oims files to label')
    parser.add_argument('--output-dir', type=str, default='labeled',
                        help='directory to write labeled extractions to')
    parser.add_argument('--binary-only', action='store_true',
                        help='run in binary classification mode instead of multi-class mode')
    parser.add_argument('-c', '--colormap', type=str, default='viridis',
                        help='colormap to use for displaying the images')
    fgroup = parser.add_mutually_exclusive_group(required=False)
    fgroup.add_argument('--focus-sun', action='store_true',
                        help='focus on data where the Sun is above the horizon')
    fgroup.add_argument('--focus-jupiter', action='store_true',
                        help='focus on low frequency (<40 MHz) data where Jupiter is above the horizon')
    args = parser.parse_args()
    if args.focus_sun or args.focus_jupiter:
        print("WARNING:  Running in focused mode - expect delays while finding suitable images")
        
    root = tk.Tk()
    app = ImageClassifierApp(root, args.data_dir, args.output_dir,
                             colormap=args.colormap,
                             is_multi_class=not args.binary_only,
                             focus_sun=args.focus_sun,
                             focus_jupiter=args.focus_jupiter)
    root.mainloop()


if __name__ == "__main__":
    main()

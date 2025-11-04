#
# Code by Felicia Brisc (University of Hamburg), distributed under a GNU General Public License v3.0
#
# Extracts 3D variables from the ICON ORCESTRA intake catalogue, remaps them to regular lon-lat and saves them to NetCDF files
#
# Run this script by providing the desired dates as command-line arguments, for example: 
# python Extract_3D_variables_to_NetCDF.py --dates 2024-08-10 2024-08-11 2024-08-12 2024-08-13
#
#
# Make sure you allocate sufficient cores on Levante — in my experience, 128 cores are required for reliable execution. 
# salloc --partition=interactive --nodes=1 -n 128 --time=12:00:00 --account your_account -- /bin/bash -c 'ssh -X $SLURM_JOB_NODELIST'
#
# Version 1.0
#

__version__ = "1.0"

import numpy as np
import xarray as xr
import intake
import subprocess
import gc
import os
import dask 
import argparse

#
# Configure dask 
def configure_dask():
        dask.config.set({
            'array.chunk-store-limit': '512MiB'  # Limit chunk size in memory
        })

        # Added this to avoid the warnings: "Warning: Slicing is producing a large chunk. To accept the large chunk and silence this warning, set the option" 
        dask.config.set({
            'array.slicing.split_large_chunks': False  
        })


# Monitor memory usage
try:
    import psutil
    def print_memory_usage(step):
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"[{step}] Memory usage: {mem_info.rss / 1024**2:.2f} MB")
except ImportError:
    def print_memory_usage(step):
        print(f"[{step}] Memory usage monitoring requires psutil. Install with: pip install psutil")


def process_date(date, dim="3d"):
    """
    Process data from the intake catalogue for a given date and dimension.
    
    Args:
        date (str): Date in YYYY-MM-DD format    
    Returns:
        None or processed data
    """
    configure_dask()
    
    output_dir = f'/your_directory_on_Levante/'
    
    try:
        print(f"Processing date: {date}")

        # Load data
        day =  date 
        print(f"Day: {day}")
        
        cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
        ds =  cat.ORCESTRA.LAM_ORCESTRA(date=day, dim=dim).to_dask()     
        print_memory_usage("After loading dataset")

        # Select variables
        n_cells = 9371648
        data_vars = ['qc', 'qi', 'pfull']     # Optionally process all: ['pfull', 'qc', 'qg', 'qi', 'qs', 'qv', 'rho', 'ta', 'ua', 'va', 'wa', 'zg']

        print(f"Variables to process: {data_vars}")
        print(f"Variable shapes: {[f'{var}: {ds[var].shape}' for var in data_vars]}")
        print(f"clat_bnds dimensions: {ds['clat_bnds'].dims}")  # Verify dimension names
        print_memory_usage("After variable selection")

        # Get coordinates         
        clat = np.degrees(ds['clat_bnds'].mean(dim='nv')).compute().values
        clon = np.degrees(ds['clon_bnds'].mean(dim='nv')).compute().values
        valid_mask = ~(np.isnan(clat) | np.isnan(clon))
        clat = clat[valid_mask]
        clon = clon[valid_mask]
        print(f"Number of valid cells: {len(clat)}")
        print_memory_usage("After computing centroids")

        # Filter to desired region
        lon_min, lon_max = -60, -10
        lat_min, lat_max = -2, 22
        region_mask = (clon >= lon_min) & (clon <= lon_max) & (clat >= lat_min) & (clat <= lat_max)
        if not np.any(region_mask):
            raise ValueError(
                f"No points in region lon: [{lon_min}, {lon_max}], lat: [{lat_min}, {lat_max}]. Check coordinates."
            )
        clat = clat[region_mask]
        clon = clon[region_mask]
        valid_cells = np.where(valid_mask)[0][region_mask]
        print(f"Number of points in region: {len(clat)}")
        print_memory_usage("After region filtering")

        # Create source grid file for CDO - in this case of 0.025 resolution, modify to your desired resolution otherwise
        source_grid_file = f'source_grid_0.025.txt'

        # Add the output directory
        source_grid_file_full_path = output_dir+source_grid_file

        with open(source_grid_file_full_path, 'w') as f:
            f.write('gridtype = unstructured\n')
            f.write(f'gridsize = {len(clat)}\n')
            f.write('xvals = ')
            np.savetxt(f, clon, fmt='%.6f', newline=' ')
            f.write('\n')
            f.write('yvals = ')
            np.savetxt(f, clat, fmt='%.6f', newline=' ')

        print(f"Created CDO source grid file: {source_grid_file_full_path}")
        print_memory_usage("After creating source grid")

        # Create CDO target grid file
        grid_res = 0.025
        lon_target = np.arange(lon_min, lon_max + grid_res, grid_res)
        lat_target = np.arange(lat_min, lat_max + grid_res, grid_res)
        nlon = len(lon_target)
        nlat = len(lat_target)
        target_grid_file = f'target_grid_0.025.txt'

        # Add the output directory:
        target_grid_file_full_path = output_dir+target_grid_file

        with open(target_grid_file_full_path, 'w') as f:
            f.write('gridtype = lonlat\n')
            f.write(f'xsize = {nlon}\n')
            f.write(f'ysize = {nlat}\n')
            f.write(f'xfirst = {lon_min}\n')
            f.write(f'xinc = {grid_res}\n')
            f.write(f'yfirst = {lat_min}\n')
            f.write(f'yinc = {grid_res}\n')

        print(f"Created CDO target grid file: {target_grid_file_full_path}")
        print_memory_usage("After creating target grid")

        # Process each variable
        for var in data_vars:
            print(f"\nProcessing variable: {var}")
            dims = ds[var].dims
            if set(dims) == {'time', 'height_full', 'cell'}:
                data_dims = ['time', 'height_full', 'cell']
                vert_coord = 'height_full'
                vert_values = ds['height_full'].values
                time_values = ds['time'].values
                has_time = True
            elif set(dims) == {'time', 'height_half', 'cell'}:
                data_dims = ['time', 'height_half', 'cell']
                vert_coord = 'height_half'
                vert_values = ds['height_half'].values
                time_values = ds['time'].values
                has_time = True
            elif set(dims) == {'height_full', 'cell'}:
                data_dims = ['height_full', 'cell']
                vert_coord = 'height_full'
                vert_values = ds['height_full'].values
                has_time = False
            else:
                print(f"Skipping {var}: unexpected dimensions {dims}")
                continue

            # Process time steps sequentially 
            if has_time:
                for t, time_val in enumerate(time_values):
                    print(f"Processing time step {t} ({time_val}) for {var}")
                    # Extract single time step
                    var_data = ds[var].isel(time=t).compute().values
                    var_data_filtered = var_data[:, valid_cells]
                    # Broadcast to 3D for correct dimensions: (1, height_full, cell)
                    var_data_3d = var_data_filtered[np.newaxis, :, :]
                    print(f"Filtered data shape for {var} time {t}: {var_data_filtered.shape} -> 3D: {var_data_3d.shape}")
                    print_memory_usage(f"After filtering {var} time {t}")

                    # Validate non-NaN values
                    non_nan_count = np.count_nonzero(~np.isnan(var_data_3d))
                    print(f"Non-NaN values in {var} time {t}: {non_nan_count} / {var_data_3d.size}")
                    if non_nan_count == 0:
                        print(f"Warning: {var} time {t} is empty. Skipping.")
                        continue

                    # Create xarray Dataset with full 3D dims for var
                    ds_unstructured = xr.Dataset(
                        {
                            var: (data_dims, var_data_3d)  # Include time dimension
                        },
                        coords={
                            'time': [time_val],  # Single time step
                            vert_coord: vert_values,
                            'cell': np.arange(len(clat)),
                            'latitude': (['cell'], clat),
                            'longitude': (['cell'], clon)
                        },
                        attrs={
                            'description': f'ICON ORCESTRA data ({var}) time step {t} on unstructured grid',
                            'gridtype': 'unstructured'
                        }
                    )

                    # Add metadata
                    ds_unstructured[var].attrs = {
                        'units': ds[var].attrs.get('units', ''),
                        'long_name': ds[var].attrs.get('long_name', var),
                        'standard_name': ds[var].attrs.get('standard_name', var),
                        'coordinates': 'latitude longitude'
                    }
                    ds_unstructured['latitude'].attrs = {
                        'units': 'degrees_north',
                        'long_name': 'Latitude',
                        'standard_name': 'latitude'
                    }
                    ds_unstructured['longitude'].attrs = {
                        'units': 'degrees_east',
                        'long_name': 'Longitude',
                        'standard_name': 'longitude'
                    }
                    ds_unstructured[vert_coord].attrs = {
                        'units': ds[vert_coord].attrs.get('units', 'm'),
                        'long_name': 'Height',
                        'standard_name': 'height'
                    }

                    # Save unstructured data
                    input_file = f'unstructured_{var}_{day}_t{t}.nc'

                    input_file_full_path = output_dir+input_file

                    ds_unstructured.to_netcdf(input_file_full_path, compute=True)
                    print(f"Saved unstructured data to {input_file_full_path}")
                    print_memory_usage(f"After saving unstructured {var} time {t}")

                    # Run CDO regridding to regular lon-lat
                    output_file = f'regridded_{var}_{day}_t{t}.nc'                    
           
                    output_file_full_path = output_dir+output_file
                    cdo_command = f"cdo -P 4 -remapnn,{target_grid_file_full_path} -setgrid,{source_grid_file_full_path} {input_file_full_path} {output_file_full_path}"

                    try:
                        subprocess.run(cdo_command, shell=True, check=True)
                        print(f"Saved regridded data to {output_file_full_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"CDO command failed for {var} time {t}: {e}")

                    # Verify output
                    ds_check = xr.open_dataset(output_file_full_path)
                    print(f"Output for {var} time {t}:\n{ds_check}")
                    non_nan_count = np.count_nonzero(~np.isnan(ds_check[var].values))
                    print(f"Non-NaN values in regridded {var} time {t}: {non_nan_count} / {ds_check[var].size}")
                    ds_check.close()
                    print_memory_usage(f"After verifying {var} time {t}")

                    # Clean up
                    del var_data, var_data_filtered, var_data_3d, ds_unstructured
                    gc.collect()
            else:
                # Process 2D variable (no time dimension)
                print(f"Processing 2D variable {var} (no time dimension)")
                var_data = ds[var].compute().values
                var_data_filtered = var_data[:, valid_cells]
                print(f"Filtered data shape for {var}: {var_data_filtered.shape}")
                print_memory_usage(f"After filtering {var}")

                # Validate non-NaN values
                non_nan_count = np.count_nonzero(~np.isnan(var_data_filtered))
                print(f"Non-NaN values in {var}: {non_nan_count} / {var_data_filtered.size}")
                if non_nan_count == 0:
                    print(f"Warning: {var} is empty. Skipping.")
                    continue

                # Create xarray Dataset
                ds_unstructured = xr.Dataset(
                    {
                        var: (data_dims, var_data_filtered)
                    },
                    coords={
                        vert_coord: vert_values,
                        'cell': np.arange(len(clat)),
                        'latitude': (['cell'], clat),
                        'longitude': (['cell'], clon)
                    },
                    attrs={
                        'description': f'ICON ORCESTRA data ({var}) on unstructured grid',
                        'gridtype': 'unstructured'
                    }
                )

                # Add metadata
                ds_unstructured[var].attrs = {
                    'units': ds[var].attrs.get('units', ''),
                    'long_name': ds[var].attrs.get('long_name', var),
                    'standard_name': ds[var].attrs.get('standard_name', var),
                    'coordinates': 'latitude longitude'
                }
                ds_unstructured['latitude'].attrs = {
                    'units': 'degrees_north',
                    'long_name': 'Latitude',
                    'standard_name': 'latitude'
                }
                ds_unstructured['longitude'].attrs = {
                    'units': 'degrees_east',
                    'long_name': 'Longitude',
                    'standard_name': 'longitude'
                }
                ds_unstructured[vert_coord].attrs = {
                    'units': ds[vert_coord].attrs.get('units', 'm'),
                    'long_name': 'Height',
                    'standard_name': 'height'
                }

                # Save unstructured data
                input_file = f'unstructured_{var}_{day}.nc'

                input_file_full_path = output_dir+input_file

                ds_unstructured.to_netcdf(input_file_full_path, compute=True)
                print(f"Saved unstructured data to {input_file}")
                print_memory_usage(f"After saving unstructured {var}")

                # Run CDO regridding
                output_file = f'regridded_{var}_{day}.nc'
                output_file_full_path = output_dir+output_file
                cdo_command = f"cdo -P 4 -remapnn,{target_grid_file_full_path} -setgrid,{source_grid_file_full_path} {input_file_full_path} {output_file_full_path}"
                try:
                    subprocess.run(cdo_command, shell=True, check=True)
                    print(f"Saved regridded data to {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f"CDO command failed for {var}: {e}")

                # Verify output
                ds_check = xr.open_dataset(output_file_full_path)
                print(f"Output for {var}:\n{ds_check}")
                non_nan_count = np.count_nonzero(~np.isnan(ds_check[var].values))
                print(f"Non-NaN values in regridded {var}: {non_nan_count} / {ds_check[var].size}")
                ds_check.close()
                print_memory_usage(f"After verifying {var}")

                # Clean up
                del var_data, var_data_filtered, ds_unstructured
                gc.collect()

        # Close dataset
        ds.close()
        print_memory_usage("After closing dataset")
        
    except Exception as e:
        print(f"Error processing {date}: {e}")
        return None

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process ORCESTRA data for multiple dates")
parser.add_argument('--dates', nargs='+', required=True, 
                    help="List of dates in YYYY-MM-DD format (e.g., 2024-08-10 2024-08-11)")
args = parser.parse_args()

# Process all dates
for date in args.dates:

    process_date(date, dim="3d")



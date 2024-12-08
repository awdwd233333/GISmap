import rasterio
import numpy as np
from .load import load
from .array2raster import array2raster
from osgeo import gdal, gdal_array, osr

def read_geo( geo ):
    proj = geo.GetProjection()
    proj = osr.SpatialReference( proj )
    GeoTransf = {
        'PixelSize' : abs( geo.GetGeoTransform()[1] ),
        'X' : geo.GetGeoTransform()[0],
        'Y' : geo.GetGeoTransform()[3],
        'Lx' : geo.RasterXSize,
        'Ly' : geo.RasterYSize
        }
    return GeoTransf, proj

def gradient( dem_file, shadingfile, Nodata=None ):
    with rasterio.open(dem_file) as src:
        elevation = src.read(1)
        transform = src.transform
    elevation = np.where(elevation <= 0, np.nan, elevation)
    print(np.nanmin(elevation), np.nanmax(elevation))
    bounds = src.bounds
    # extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    _, Geotrans, proj = load(dem_file)
    grad_x, grad_y = np.gradient(elevation)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    array2raster(gradient_magnitude, shadingfile, proj=proj, GeoTransf=Geotrans, NoDataValue=Nodata)
    return gradient_magnitude

def dem_lt( dem_file, value, clip_dem_file, Nodata ):
    with rasterio.open(dem_file) as src:
        elevation = src.read(1)
        transform = src.transform
    elevation = np.where(elevation <= 0, np.nan, elevation)
    elevation = np.where(elevation <= value, elevation, np.nan)
    bounds = src.bounds
    # extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    _, Geotrans, proj = load(dem_file)
    array2raster( elevation, clip_dem_file, proj=proj, GeoTransf=Geotrans, NoDataValue=Nodata)
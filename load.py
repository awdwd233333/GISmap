from __future__ import division
import os, pickle, time, calendar, sys, datetime, warnings, operator, logging
from osgeo import gdal, gdal_array, osr
try:
	import ogr
except ModuleNotFoundError:
	from osgeo import ogr
import numpy as np
import shapefile
import os
try:
	import ConfigParser
except ImportError:
	import configparser as ConfigParser
from osgeo import gdal, gdal_array, osr

def load( fname, *args, **kwargs ):
    from xml.dom.minidom import parse
    '''Load file depending on the extension'''
    if fname is None:return None
    ext = os.path.splitext( fname )[-1]

    if not os.path.isfile(fname):return None

    if ext == '.txt':
        return np.loadtxt( fname, *args, **kwargs )
    elif ext == '.npy':
        arr = np.load( fname, *args, **kwargs )
        if len(arr.shape) == 3 and arr.shape[-1]==1:
            arr = arr[:,:,0]
        return arr
    elif ext == '.p':
        return pickle.load( open(fname,'rb') )
    elif ext == '.prj':
        proj = osr.SpatialReference()
        fn = open(fname,'r')
        Wkt=fn.read()
        fn.close()
        proj.ImportFromWkt(Wkt)
        return proj
    elif ext in ['.kml','.xml']:
        # 目前只能读面的信息
        doc = parse( fname )
        root = doc.documentElement
        coordinates = root.getElementsByTagName("coordinates")
        obj_list=[]
        for coordinate in coordinates:
            old_data = coordinate.childNodes[0].data
            new_data = " ".join([old.replace(",", " ") for old in old_data.split(",0")])
            new_data = new_data.strip('\n').strip('\t').strip(' ')
            #t = [pnt for pnt in new_data.split('  ') if (pnt!=[''] and pnt!=['\n'])]
            pnts=[[float(i.strip(' ').strip('\t').strip('\n')) for i in pnt.split(' ')] for pnt in new_data.split('  ') if pnt!='\n']
            obj_list.append( pnts )
        return obj_list
    elif ext in ['.shp']:
        return [shape.points for shape in shapefile.Reader( fname ).shapes()]
    elif ext == '.ini':
        cf = ConfigParser()
        cf.read( fname )
        return cf
    elif ext in ['.tif','.tiff']:
        from .raster import read_geo
        arr = gdal_array.LoadFile( fname )
        geo = gdal.Open( fname )
        Geotrans, proj = read_geo(geo)
        return arr, Geotrans, proj
    else:
        e = 'Format %s not supported for file %s. Use either "npy" or "txt"' % ( ext, fname )
        raise TypeError(e)


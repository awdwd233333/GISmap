from osgeo import gdal, gdal_array, osr
import numpy as np

def array2raster( in_array, raster_path, band_num=None, proj=None, GeoTransf=None, NoDataValue=None ):
    # in_array是2/3维数组,层数在第三维,proj是 osr.SpatialReference()
    # band_num是类似[0,2,3],最小是0

    # 判断栅格数据的数据类型
    if 'int8' in in_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in in_array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName("GTiff")
    #outdata = driver.Create(raster_path,ysize=in_array.shape[0],xsize=in_array.shape[1],bands=len(band_num),datatype=datatype)
    if len(in_array.shape) == 2:in_array = in_array[:,:,np.newaxis]
    if band_num is None:band_num = range( 0, in_array.shape[2] )
    outdata = driver.Create( raster_path, in_array.shape[1], in_array.shape[0], len(band_num), datatype )
    for i,idx in enumerate( band_num ):
        outband = outdata.GetRasterBand( i + 1 )
        in_array_band = in_array[:,:,idx]
        outband.WriteArray( in_array_band )
        if NoDataValue is not None:outband.SetNoDataValue(NoDataValue)

    if GeoTransf != None:
        pcsize = GeoTransf['PixelSize']
        originX, originY = GeoTransf['X'], GeoTransf['Y']
        outdata.SetGeoTransform( (originX, pcsize, 0, originY, 0, -pcsize) )
        # geotrans的标准格式 [x,width,0,y,0,-height]
    if proj != None:
        outdata.SetProjection( proj.ExportToWkt() )
    outdata.FlushCache()
    outdata = None


def pntsTrans( lines, EPSG, EPSG_out ):
    from pyproj import CRS, Transformer
    # lines为经纬度或者xy, 输出为经纬度或者xy
    # 把 xy 地理坐标转化为投影坐标，gge默认地理坐标是WGS84
    transformer = Transformer.from_crs( CRS.from_epsg(EPSG), CRS.from_epsg(EPSG_out) )
    n_lines=[]
    for line in lines:
        if EPSG==4326:y, x = list(zip(*line))
        else:x, y = list(zip(*line))
        [x, y] = transformer.transform( x, y )
        if EPSG_out==4326:x,y=y,x
        n_lines.append(list(zip(x,y)))
    return n_lines

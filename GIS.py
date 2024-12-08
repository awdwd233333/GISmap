"""
@Name: GIS.py
@Auth: HeYucong (hyc0424@whu.edu.cn, 18188425393)
@Date: 2024/10/20-22:35
@Link: https://blog.csdn.net/awdwd233333
@Desc: 
@Ver : 0.0.0
"""
import os, matplotlib, rasterio, geemap
import numpy as np
import pandas as pd
import cartopy.feature as cfeature
from shapely.ops import unary_union
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from math import floor
from matplotlib import patheffects
from matplotlib.colors import Normalize
from rasterio.plot import show
import cmocean, cartopy, ee
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import geopandas as gpd
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
from matplotlib import font_manager as fm
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch

plt.rcParams['axes.unicode_minus']=False#显示负号
plt.rcParams["font.size"]=10#全局字号,针对英文
plt.rcParams["mathtext.fontset"]='stix'#设置$$之间是字体与中文不一样,为新罗马,这样就可以中英混用
plt.rcParams['font.sans-serif'] = ['Times New Roman']#单独使用英文时为新罗马字体

# if os.name == 'nt':
#     matplotlib.rc('font', family='Arial')
# else:  # might need tweaking, must support black triangle for N arrow
#     matplotlib.rc('font', family='DejaVu Sans')


def draw_Mapring(ax, df, v_names, max_values=None, size=1, semi_circle=False, width=0.15, legend_size=3,
                 legend_pos=(100, 34), legend_width=0.1, units=None, r0=.4, arrow_factor=0.4, label_gap=None,
                 colors=None, panel_width=0.3):
	'''
	v_names:list of data name to show
	'''
	if max_values is None:
		max_values = []
		for v_name in v_names:
			max_values += [abs(df[v_name]).max()]

	for _, row in df.iterrows():
		values = row[v_names]
		if type(row['pnt']) != str: continue
		pnt = eval(row['pnt'])
		ax_ = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(pnt[1]-size/2, pnt[0]-size/2, size, size),
		                 bbox_transform=ax.transData, loc='center')  # borderpad=0,
		draw_ring(ax_, values, max_values, semi_circle=semi_circle, width=width, r0=r0, colors=colors)

	ax_ = inset_axes(ax, width="100%", height="100%",
	                 bbox_to_anchor=(legend_pos[0], legend_pos[1], legend_size, legend_size),
	                 bbox_transform=ax.transData, loc='lower center')  # borderpad=0,
	wedge = Wedge((0.5, 0.5), r0, -90, 90, width=None, color='white')
	ax_.add_patch(wedge)
	draw_ring(ax_, max_values, max_values, semi_circle=semi_circle, width=legend_width, r0=r0, colors=colors)
	rect = plt.Rectangle((0.5 - panel_width, 0.1), width=panel_width, height=0.8, ec='none', fc='white', angle=0, fill=True, alpha=1, transform=ax_.transAxes)
	ax_.add_patch(rect)

	if units is None: units = [''] * len(v_names)
	if label_gap is None: label_gap = legend_width
	for idx, (v_name, max_value, unit) in enumerate(zip(v_names, max_values, units)):
		ax_.text(0.5, 0.9 - idx * label_gap, s=v_name + ' ', fontsize=6, color="k", transform=ax_.transAxes, rotation=0,
		         ha='right', va='top')
		ax_.text(0.5, 0.1 + idx * label_gap, s='%.3g %s' % (max_value, unit), fontsize=6, color="k",
		         transform=ax_.transAxes, rotation=0, ha='right', va='bottom')

	edge_width = 1
	arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(2 * edge_width, 3 * edge_width, edge_width)
	y = 0.5 + r0 - legend_width * len(v_names) * (1 + arrow_factor)
	x1 = 0.5 + legend_width * len(v_names) * (1 - arrow_factor)
	x2 = 0.5 - legend_width * len(v_names) * (1 - arrow_factor)

	arrow = FancyArrowPatch(posA=(0.5, y), posB=(x1, 0.5), arrowstyle=arrowstyle, color='k',
	                        connectionstyle="arc3,rad=-0.5", shrinkA=0, shrinkB=0)
	ax_.add_artist(arrow)
	arrow = FancyArrowPatch(posA=(0.5, y), posB=(x2, 0.5), arrowstyle=arrowstyle, color='k',
	                        connectionstyle="arc3,rad=0.5", shrinkA=0, shrinkB=0)
	ax_.add_artist(arrow)
	ax_.text((0.5 + x1) / 2, (0.5 * 2 + y) / 3, s=r'+', fontsize=6, color="k", transform=ax_.transAxes, rotation=0, fontweight='bold',
	         ha='center', va='center')
	ax_.text((0.5 + x2) / 2, (0.5 * 2 + y) / 3, s=r'-', fontsize=6, color="k", transform=ax_.transAxes, rotation=0, fontweight='bold',
	         ha='center', va='center')


def draw_ring( ax, values, max_values, single_max=None, semi_circle=False, width=0.1, r0 = 0.48, colors=None ):
	'''
	Draw single/multiple ring within a ax with its/their values
	:param ax:
	:param values:list,
	:param max_v:list,
	:param semi_circle:
	:param width:
	:param r0:inner radius of ring
	:return:
	'''
	from matplotlib.patches import Wedge
	if single_max is not None:max_values = [single_max]*len(values)
	if not semi_circle:max_v = [v/2 for v in max_values]

	if len(values)==1:
		value = values[0]
		if value >= 0:colors = ['g']
		elif value < 0:colors = ['r']
	if colors is None:colors = [ plt.get_cmap('tab10')(i/10) for i in range(10) ][:len(values)]

	# if len(values)!=1:
	# 	wedge = Wedge((0.5, 0.5), r0, 0, 360, width=None, color='white')
	# 	ax.add_patch(wedge)

	for idx,(value,max_v,color) in enumerate(zip(values,max_values,colors)):
		if value >= 0:
			if semi_circle:
				thata = (90-value/max_v*180,90)
				wedge = Wedge((0.5, 0.5), r0 - idx * width * 1.1, -90, 90, width=width, color='white')
			else:
				thata = (90 - value / max_v * 360, 90)
				wedge = Wedge((0.5, 0.5), r0-idx*width*1.1, 0, 360, width=width, color='white')
		elif value < 0:
			if semi_circle:
				thata = (90,90-value/max_v*180)
				wedge = Wedge((0.5, 0.5), r0 - idx * width * 1.1, 90, -90, width=width, color='white')
			else:
				thata = (90 - value / max_v * 360, 90)
				wedge = Wedge((0.5, 0.5), r0-idx*width*1.1, 0, 360, width=width, color='white')
		ax.add_patch(wedge)
		wedge = Wedge( (0.5,0.5), r0-idx*width*1.1, thata[0], thata[1], width=width, color=color)
		ax.add_patch(wedge)
	# if len(values)!=1:
	# 	wedge = Wedge((0.5, 0.5), r0 - (idx+1) * width * 1.1, 0, 360, width=None, color='white')
	# 	ax.add_patch(wedge)

	ax.set_facecolor('none')
	ax.axis('off')
	ax.set_aspect('equal', adjustable='box')


def draw_dem( ax, file_path, elevation_thresh=None, cmap='terrain', unit='km', cax_gap=0.05, cax_width=0.02,fontsize=10, vmin=None, vmax=None, nodata=None, extent=None ):
	with rasterio.open(file_path) as src:
		elevation = src.read(1)
		transform = src.transform
	if nodata is not None:elevation = np.where(elevation == nodata, np.nan, elevation)
	elevation = np.where(elevation <= 0, np.nan, elevation)

	# 获取地理边界
	bounds = src.bounds
	extent_img = [bounds.left, bounds.right, bounds.bottom, bounds.top]

	if extent is None:
		extent = extent_img
		if extent[1] - extent[0] < 3:
			ax.set_xticks(np.linspace(extent[0], extent[1], 3).round(1), crs=ccrs.PlateCarree())
		else:
			ax.set_xticks(np.linspace(extent[0], extent[1], 3).astype(int), crs=ccrs.PlateCarree())
		if extent[3] - extent[2] < 3:
			ax.set_yticks(np.linspace(extent[2], extent[3], 3).round(1), crs=ccrs.PlateCarree())
		else:
			ax.set_yticks(np.linspace(extent[2], extent[3], 3).astype(int), crs=ccrs.PlateCarree())

		ax.set_xlim(extent[:2])
		ax.set_ylim(extent[-2:])
		ax.set_extent(extent, crs=ccrs.PlateCarree())
		# 创建图形和子图
	platecarree = ccrs.PlateCarree(globe=ccrs.Globe(datum='WGS84'))
	# 设置地图范围

	grad_x, grad_y = np.gradient(elevation)
	gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
	shade_img = ax.imshow( gradient_magnitude, extent=extent_img, transform=ccrs.PlateCarree(),cmap='gray_r', alpha=1, vmin=-50, vmax=600 )
	if elevation_thresh is not None:
		elevation = np.where(elevation > elevation_thresh, elevation, np.nan)
	if vmin is None:vmin = np.nanmin(elevation)
	if vmax is None:vmax = np.nanmax(elevation)
	norm = Normalize(vmin=np.nanmin(elevation), vmax=np.nanmax(elevation))
	img = ax.imshow(elevation, extent=extent_img, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

	ax_position = ax.get_position()
	cax = ax.get_figure().add_axes([ax_position.x0 + ax_position.width + cax_gap, ax_position.y0, cax_width, ax_position.height])
	if unit == 'km':
		func = lambda x, pos: "{:g}".format(x / 1000)
	elif unit == 'm':
		func = lambda x, pos: "{:g}".format(x)
	fmt = matplotlib.ticker.FuncFormatter(func)
	cbar = ax.get_figure().colorbar(img, cax=cax, orientation='vertical', pad=0.05, aspect=10, extend='both', label='', format=fmt)
	cbar.set_label('Elevation / %s' % unit, fontsize=fontsize)
	cax.tick_params(axis='both', which='major', labelsize=fontsize, rotation=0, color='black', labelcolor='black', colors='black', width=1, length=2, pad=3, direction='in', top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelright=False)

	ax.coastlines(resolution='110m', linewidth=0.2)  # 添加海岸线和地理特征
	# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
	# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	# city = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))#'naturalearth_lowres'
	# world.plot(
	#     ax=ax,
	#     color="lightgray",
	#     edgecolor="black",
	#     alpha=0.5
	# )
	# city.plot(
	#     ax=ax,
	#     color="lightgray",
	#     edgecolor="black",
	#     alpha=0.5
	# )

	# 添加经纬刻度
	from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,	                                LatitudeLocator, LongitudeLocator)
	# ax.yaxis.tick_right()
	lon_formatter = LongitudeFormatter(zero_direction_label=True)
	lat_formatter = LatitudeFormatter()
	ax.xaxis.set_major_formatter(lon_formatter)
	ax.yaxis.set_major_formatter(lat_formatter)
	ax.tick_params(axis='both', which='major', labelsize=10, rotation=0, color='black', labelcolor='black', colors='black', width=1, length=4, pad=2, direction='in', top=False, bottom=True, left=True,right=False, labelleft=True, labelbottom=True)
	ax.tick_params(axis='y', rotation=90)
	# 添加经纬刻网
	# gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True, linewidth=1, color='k', alpha=0, linestyle='-')
	# gl.xlines = False
	# gl.ylines = False
	# gl.top_labels = False
	# gl.right_labels = False
	# gl.xlocator = plt.FixedLocator(np.arange(int(bounds.left), int(bounds.right) + 1, 4))
	# gl.ylocator = plt.FixedLocator(np.arange(int(bounds.bottom), int(bounds.top) + 1, 5))
	# gl.xlabel_style = {'size': 8, 'color': 'k'}
	# gl.ylabel_style = {'size': 8, 'color': 'k'}
	# ax.set_xticks([])
	# ax.set_yticks([])
	# ax.set_title("Basic Map of World with GeoPandas")

	ax.add_feature(cfeature.LAKES.with_scale('50m'))
	# ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5)
	ax.add_feature(cfeature.OCEAN, linewidth=0.5)
	# ax.add_feature(cfeature.nightshade.Nightshade)

def draw_stream_basin( ax, Basin, R1=None, R2=None, R3=None, R0=None, extent=None, draw_basin=True, vis_param={} ):
	import geopandas as gpd
	import cartopy.feature as cfeature
	ee.Initialize()
	# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	city = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))#'naturalearth_lowres'

	# world.plot(
	#     ax=ax,
	#     color="lightgray",
	#     edgecolor="black",
	#     alpha=0.5
	# )
	# city.plot(
	# 	ax=ax,
	# 	color="lightgray",
	# 	edgecolor="black",
	# 	alpha=0.5
	# )
	if Basin is not None:
		if not Basin.empty:
			color = vis_param.get('basin_color',"gray")
			edgecolor = vis_param.get('basin_edgecolor',"none")
			if draw_basin:Basin.plot( ax=ax,color=color,alpha=0.5, edgecolor=edgecolor )
			if extent is None:
				bounds = Basin.bounds
				extent = [bounds.minx[0], bounds.maxx[0], bounds.miny[0], bounds.maxy[0]]

			if extent[1] - extent[0] < 3:
				ax.set_xticks(np.linspace(extent[0], extent[1], 3).round(1), crs=ccrs.PlateCarree())
			else:
				ax.set_xticks(np.linspace(extent[0], extent[1], 3).astype(int), crs=ccrs.PlateCarree())
			if extent[3] - extent[2] < 3:
				ax.set_yticks(np.linspace(extent[2], extent[3], 3).round(1), crs=ccrs.PlateCarree())
			else:
				ax.set_yticks(np.linspace(extent[2], extent[3], 3).astype(int), crs=ccrs.PlateCarree())
			ax.set_xlim(extent[:2])
			ax.set_ylim(extent[-2:])

	region = ee.Geometry({'type': 'Polygon', 'coordinates': [[(extent[0], extent[2]), (extent[0], extent[3]), (extent[1], extent[3]), (extent[1], extent[2]), (extent[0], extent[2])]]})
	if R0 is not None:
		gdf_region = geemap.ee_to_gdf(ee.FeatureCollection(region))
		if len(Basin.geometry.tolist()) == 1:
			R0 = R0.clip(gdf_region.difference(Basin.geometry.tolist()[0]))
		else:
			R0 = R0.clip(gdf_region.difference(MultiPolygon(Basin.geometry.tolist())))
		color = vis_param.get('R0_color', "#98b7e1")
		linestyle = vis_param.get('R0_linestyle', '-')
		R0.plot( ax=ax, color=color, linewidth=1,alpha=1,markersize=10, marker=',', linestyle=linestyle )
	if R1 is not None:
		R1 = R1.clip(Basin)
		color = vis_param.get('R1_color', "b")
		linestyle = vis_param.get('R1_linestyle', '-')
		R1.plot(ax=ax, color=color, linewidth=1,alpha=1,markersize=10, marker=',', linestyle=linestyle )
	if R2 is not None:
		color = vis_param.get('R2_color', "b")
		linestyle = vis_param.get('R2_linestyle', '-')
		R2.plot(ax=ax, color=color, linewidth=0.5, alpha=1, markersize=10, marker=',', linestyle=linestyle )
	if R3 is not None:
		color = vis_param.get('R2_color', "b")
		linestyle = vis_param.get('R2_linestyle', '-')
		R3 = R3.set_geometry("geometry")
		R3.plot(ax=ax, color=color, linewidth=0.2, alpha=1, markersize=10, marker=',', linestyle=linestyle )

	# 添加经纬刻度
	from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,	                                LatitudeLocator, LongitudeLocator)
	# ax.yaxis.tick_right()
	lon_formatter = LongitudeFormatter(zero_direction_label=True)
	lat_formatter = LatitudeFormatter()
	ax.xaxis.set_major_formatter(lon_formatter)
	ax.yaxis.set_major_formatter(lat_formatter)
	ax.tick_params(axis='both', which='major', labelsize=10, rotation=0, color='black', labelcolor='black', colors='black', width=1, length=4, pad=2, direction='in', top=False, bottom=True, left=True,right=False, labelleft=True, labelbottom=True)
	ax.tick_params(axis='y', rotation=90)

	ax.add_feature(cfeature.LAKES.with_scale('50m'))
	if R1 is None:ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5)
	ax.add_feature(cfeature.OCEAN, linewidth=0.5)



def eeRiver_to_gdf(Network, shrink=True):
	R = Network.toList(1e6)
	length = R.length().getInfo()
	if length>5e3:
		if shrink:
			while length > 5000:
				thresh = Network.aggregate_min('DIS_AV_CMS').getInfo() * 2
				Network = Network.filter(ee.Filter.gt('DIS_AV_CMS', thresh))
				length = Network.toList(1e6).length().getInfo()
			gdf = geemap.ee_to_gdf(Network)
		else:
			col = []
			gdf = gpd.GeoDataFrame()
			for j in range(int(length//5e3+1)):
				if j==len(range(int(length//5e3+1)))-1:
					end = length
				else:end = (j+1)*5000
				Network = ee.FeatureCollection([R.get(i) for i in range(j*5000,end)])
				gdf = gdf.append( geemap.ee_to_gdf(Network) )
	else:
		gdf = geemap.ee_to_gdf(Network)
	return gdf

def ee_to_gdf( Fc ):
	R = Fc.toList(1e6)
	length = R.length().getInfo()
	if length>5e3:
		col = []
		gdf = gpd.GeoDataFrame()
		for j in range(int(length//5e3+1)):
			if j==len(range(int(length//5e3+1)))-1:
				end = length
			else:end = (j+1)*5000
			Fc = ee.FeatureCollection([R.get(i) for i in range(j*5000,end)])
			gdf = gdf.append( geemap.ee_to_gdf(Fc) )
	else:
		gdf = geemap.ee_to_gdf(Fc)
	return gdf

def Hydro_map( ax, ROI, Basin_level=4, Stream_level=None, shrink=False, DEM=True, res=None, shade_lim=[-50,600], gdf_pos=None, draw_basin=True, extent=None, stream_n=3, draw_scaler=True, draw_N=True, vis_param={}, draw_outer_stream=True, Basin=None, Basindir=r'.', elevation_thresh=None, Lake_sat=None, draw_dam = False ):
	'''

	:param ax:
	:param ROI:
	:param Basin_level:
	:param Stream_level:
	:param DEM:
	:param res:
	:param shade_lim:
	:param gdf_pos:
	:param draw_basin:
	:param extent:
	:param stream_n:
	:param draw_scaler:
	:param draw_N:
	:param vis_param:
	:param draw_outer_stream:
	:param Basin: customed fc collection
	:return:
	'''
	import os, geemap, ee
	import numpy as np
	from .load import load
	from .basic import scale_bar, find_position,N_arrow
	ee.Initialize()

	if ax is None:
		fig = plt.figure(figsize=(5.5, 4))
		ax = fig.add_axes([0.08, 0.2, 0.85, 0.8], projection=ccrs.PlateCarree())
		show=True
	else:
		show=False

	if not os.path.isdir( Basindir ): os.mkdir( Basindir )

	if Basin is None:
		WWF = ee.FeatureCollection("WWF/HydroATLAS/v1/Basins/level0%s"%Basin_level)
		Basin = WWF.filterBounds(ROI)
		gdf_Basin = geemap.ee_to_gdf(Basin)
		WWF0 = ee.FeatureCollection( "WWF/HydroATLAS/v1/Basins/level0%s" %(Basin_level-1) )
		Basin0 = WWF0.filterBounds(ROI)
		try:gdf_Basin0 = geemap.ee_to_gdf(Basin0)
		except:
			Basin0 = None
			gdf_Basin0 = None
		WWF_1 = ee.FeatureCollection( "WWF/HydroATLAS/v1/Basins/level0%s" %(Basin_level-2) )
		Basin_1 = WWF_1.filterBounds(ROI)
		try:gdf_Basin_1 = geemap.ee_to_gdf(Basin_1).set_geometry("geometry")
		except:
			Basin_1 = None
			gdf_Basin_1 = None

	if extent is None:
		bounds = gdf_Basin.bounds
		extent = [bounds.minx.min(), bounds.maxx.max(), bounds.miny.min(), bounds.maxy.max()]
	if extent[1]-extent[0]<3:
		ax.set_xticks(np.linspace(extent[0], extent[1], 3).round(1), crs=ccrs.PlateCarree())
	else:ax.set_xticks(np.linspace(extent[0], extent[1], 3).astype(int), crs=ccrs.PlateCarree())
	if extent[3] - extent[2] < 3:
		ax.set_yticks(np.linspace(extent[2], extent[3], 3).round(1), crs=ccrs.PlateCarree())
	else:
		ax.set_yticks(np.linspace(extent[2], extent[3], 3).astype(int), crs=ccrs.PlateCarree())
	ax.set_xlim(extent[:2])
	ax.set_ylim(extent[-2:])

	if draw_scaler:
		# -----------set scaler------------
		x0, x1, y0, y1 = ax.get_extent(ccrs.PlateCarree().as_geodetic())
		utm = ccrs.UTM(int(floor((x0+x1)/2/6) + 31))
		x0, x1, y0, y1 = ax.get_extent(utm)
		if int((x1 - x0)/4/1000/10)<1:# within 10 km
			length = int((x1 - x0)/4/1000)
		elif int((x1 - x0)/4/1000/10)>=1:
			length = int((x1 - x0)/4/1000/10)*10
		bbox_scaler = scale_bar( ax, ccrs.PlateCarree(), length, location=(-0.5,-0.5), linewidth=6, units='km', m_per_unit=1000,fontsize=13 )
		shape = [bbox_scaler[1]-bbox_scaler[0], bbox_scaler[3]-bbox_scaler[2]]

		pos, box_scaler = find_position( ax, gdf_Basin.geometry[0], shape, x_bias = shape[1] * 0.4, y_bias = shape[1]*0.2 )
		pos = [(pos[0]-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]),(pos[1]-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])]
		bbox_scaler = scale_bar( ax, ccrs.PlateCarree(), length, location=pos, linewidth=6, units='km', m_per_unit=1000, fontsize=13 )

	if draw_N:
		#------------set N arrow------------
		bbox_N = N_arrow( ax, ccrs.PlateCarree(), location=(0,-5), fontsize=12 )
		shape = [bbox_N[1]-bbox_N[0], bbox_N[3]-bbox_N[2]]
		box_Polygon_of_scaler = Polygon([(box_scaler[0],box_scaler[2]),(box_scaler[0],box_scaler[3]),(box_scaler[1],box_scaler[3]),(box_scaler[1],box_scaler[2])])
		pos,box_N = find_position( ax, unary_union([gdf_Basin.geometry[0],box_Polygon_of_scaler]), shape, x_bias = shape[0]*0.3, y_bias = shape[1]*0.2 )

		pos = [(pos[0]-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]),(pos[1]-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])]
		bbox_N = N_arrow( ax, ccrs.PlateCarree(), location=pos, fontsize=12 )

		xlim0, xlim1 = ax.get_xlim()
		ylim0, ylim1 = ax.get_ylim()
		extent = [xlim0, xlim1, ylim0, ylim1]
	print('extent',extent)
	region = ee.Geometry({'type': 'Polygon', 'coordinates': [[(extent[0], extent[2]), (extent[0], extent[3]), (extent[1], extent[3]), (extent[1], extent[2]), (extent[0], extent[2])]]})

	#-------Get stream network & basin data-----------
	af_river = ee.FeatureCollection("projects/sat-io/open-datasets/GRN/af_river")
	as_river = ee.FeatureCollection("projects/sat-io/open-datasets/GRN/as_river")
	au_river = ee.FeatureCollection("projects/sat-io/open-datasets/GRN/au_river")
	eu_river = ee.FeatureCollection("projects/sat-io/open-datasets/GRN/eu_river")
	na_river = ee.FeatureCollection("projects/sat-io/open-datasets/GRN/na_river")
	sa_river = ee.FeatureCollection("projects/sat-io/open-datasets/GRN/sa_river")
	# GRN = as_river.merge(af_river).merge(af_river).merge(au_river).merge(eu_river).merge(na_river).merge(sa_river)
	# GRN_region = GRN.filterBounds(region)
	# GRN_Basin = GRN_region.filterBounds(Basin.geometry().buffer(-1e4))
	if Basin_1 is not None:
		hydroriver = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(Basin_1.geometry())
	elif Basin0 is not None:
		hydroriver = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(Basin0.geometry())
	else:
		hydroriver = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(Basin.geometry())
	River_Basin = hydroriver.filterBounds(Basin.geometry())#.buffer(-1e3)
	if Stream_level is None:
		for Stream_level in range(1,10):
			Network = River_Basin.filter(ee.Filter.eq('RIV_ORD', Stream_level))# union to save a major stream
			gdf_R1 = eeRiver_to_gdf(Network, shrink=shrink)
			if not gdf_R1.empty: break
			if gdf_R1.empty: continue
			if gdf_R1.geometry[0].type=='MultiPoint':continue
	else:
		Network = River_Basin.filter(ee.Filter.eq('RIV_ORD', Stream_level))
		gdf_R1 = geemap.ee_to_gdf(Network)

	if draw_outer_stream:
		Network = hydroriver.filter(ee.Filter.lte('RIV_ORD', Stream_level)).filter(ee.Filter.gte('RIV_ORD', Stream_level - 1))
		gdf_R0 = eeRiver_to_gdf(Network, shrink=shrink).set_geometry("geometry")
		if gdf_Basin_1 is not None:
			if len(gdf_Basin.geometry.tolist())==1:
				gdf_R0 = gdf_R0.clip(gdf_Basin_1.difference(gdf_Basin.geometry.tolist()[0]))
			else:
				gdf_R0 = gdf_R0.clip(gdf_Basin_1.difference(MultiPolygon(gdf_Basin.geometry.tolist())))
		elif gdf_Basin0 is not None:
			if len(gdf_Basin.geometry.tolist())==1:
				gdf_R0 = gdf_R0.clip(gdf_Basin0.difference(gdf_Basin.geometry.tolist()[0]))
			else:
				gdf_R0 = gdf_R0.clip(gdf_Basin0.difference(MultiPolygon(gdf_Basin.geometry.tolist())))
		if gdf_R0.empty: gdf_R0 = None
	else:
		gdf_R0 = None
	if stream_n >= 2:
		Network = River_Basin.filter(ee.Filter.eq('RIV_ORD', Stream_level + 1))
		gdf_R2 = eeRiver_to_gdf(Network, shrink=shrink).set_geometry("geometry")
		if stream_n == 2: gdf_R3 = None
	if stream_n >= 3:
		Network = River_Basin.filter(ee.Filter.eq('RIV_ORD', Stream_level + 2))
		gdf_R3 = eeRiver_to_gdf(Network, shrink=shrink).set_geometry("geometry")
	else:
		gdf_R2 = None
		gdf_R3 = None
	if gdf_R1.empty:
		gdf_R1 = gdf_R2
		gdf_R2 = gdf_R3

	if gdf_R1 is not None:
		if gdf_R1.empty: gdf_R1 = None
	if gdf_R2 is not None:
		if gdf_R2.empty: gdf_R2 = None
	if gdf_R3 is not None:
		if gdf_R3.empty: gdf_R3 = None

	#---------------set dem ---------------
	if DEM:
		nodata = -999
		shadingfile = os.path.join(Basindir, 'shading.tif')
		dem_region_file = os.path.join( Basindir, 'dem_region.tif' )
		dem_Basin_file = os.path.join(Basindir, 'dem_Basin.tif')
		if (extent[3]+extent[2])/2>60:
			dem_region = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic").clip(region)
		else:
			dem_region = ee.Image("CGIAR/SRTM90_V4").clip(region)
		dem_Basin = dem_region.clip(Basin.geometry())

		if res is None:
			mask = dem_region.select('elevation').lt(1000000)
			cnt = mask.reduceRegion(
				reducer=ee.Reducer.sum(),
				geometry=Basin,
				scale=1e3,
				maxPixels=1e13).getInfo()['elevation']
			# dem = dem.reproject( scale=res)
			res = int((cnt*(1e3)**2/5e5)**0.5)
			print('cnt',cnt,'res',res)

		geemap.ee_export_image( dem_region, dem_region_file, scale=res, crs='EPSG:4326', crs_transform=None, region=region, dimensions=None, file_per_band=False, format='ZIPPED_GEO_TIFF', unmask_value=nodata, timeout=300, proxies=None)
		geemap.ee_export_image( dem_Basin, dem_Basin_file, scale=res, crs='EPSG:4326', crs_transform=None, region=Basin.geometry(), dimensions=None, file_per_band=False, format='ZIPPED_GEO_TIFF', unmask_value=nodata,
		 timeout=300, proxies=None)
		# --------shading-----------
		from .raster import gradient, dem_lt
		gradient_magnitude = gradient( dem_region_file, shadingfile )
		shade_img = ax.imshow( gradient_magnitude, extent=extent, transform=ccrs.PlateCarree(), cmap='gray_r', alpha=1, vmin=shade_lim[0], vmax=shade_lim[1] )

		# -----------draw_basin dem--------------
		if elevation_thresh is not None:
			dem_clip_file = os.path.join( Basindir, 'dem_Basin_%sm.tif'%(int(elevation_thresh)) )
			dem_lt( dem_Basin_file, elevation_thresh, dem_clip_file, Nodata=nodata )
			draw_dem( ax, dem_clip_file, cmap='terrain', unit='km', cax_gap=0.05, cax_width=0.02, fontsize=10, vmin=None, vmax=None, nodata=nodata, extent=extent )
		else:
			draw_dem( ax, dem_Basin_file, cmap='terrain', unit='km', cax_gap=0.05, cax_width=0.02,fontsize=10, vmin=None, vmax=None, nodata=nodata, extent=extent )
		draw_basin=False
	#-----------draw stream and basin-----------
	print('draw stream and basin')
	draw_stream_basin( ax, gdf_Basin, R1 = gdf_R1, R2=gdf_R2, R3=gdf_R3, R0=gdf_R0, extent=extent, draw_basin=draw_basin, vis_param=vis_param )

	#----------draw lake-----------------
	if Lake_sat is not None:
		print('draw lake')
		if Lake_sat == 'HydroLake':
			print('use HydroLake')
			HydroLakes = ee.FeatureCollection("projects/sat-io/open-datasets/HydroLakes/lake_poly_v10").filterBounds(region)
			gdf_LAKES = ee_to_gdf(HydroLakes)
			gdf_LAKES.set_crs(epsg=4326, inplace=True)
			gdf_LAKES.to_file(r'%s\LAKES.shp' % (Basindir))
			gdf_LAKES.plot(ax=ax, color='#97b6e1', linewidth=1, alpha=1, markersize=10, marker=',', linestyle='-')
		# ee.FeatureCollection("projects/sat-io/open-datasets/ReaLSAT/ReaLSAT-1_4").filterBounds(region)#.filter(ee.Filter.gt('AREA', 5e4))
		elif Lake_sat == 'JRC':
			print('use JRC')
			dem_water_file = os.path.join(Basindir, 'dem_water.tif')
			water = ee.Image(ee.ImageCollection("JRC/GSW1_4/YearlyHistory").filter(ee.Filter.calendarRange(2021,2021,'year')).first()).eq(ee.Image.constant(3)).clip(region)# permanant water
			geemap.ee_export_image( water, dem_water_file, scale=res, crs=None, crs_transform=None, region=region,dimensions=None, file_per_band=False, format='ZIPPED_GEO_TIFF', unmask_value=0,timeout=300, proxies=None)
			water = load(dem_water_file)[0]
			water = np.where(water<=0,np.nan,1)
			ax.imshow( water, extent=extent, transform=ccrs.PlateCarree(), cmap=matplotlib.colors.ListedColormap(['#97b6e1']))
	else:
		gdf_LAKES = gpd.GeoDataFrame([[0]],columns=['idx'])
		gdf_LAKES['geometry'] = MultiPolygon(list(cfeature.LAKES.with_scale('50m').geometries()))
		gdf_LAKES.set_crs(epsg=4326, inplace=True)
		gdf_LAKES.to_file(r'%s\LAKES.shp' % (Basindir))
	if draw_dam:
		print('draw_dam')
		dams = ee.FeatureCollection("projects/sat-io/open-datasets/GOODD/GOOD2_dams").filterBounds(Basin)
		df_dams = ee_to_gdf(dams)
		if not df_dams.empty:
			df_dams = df_dams.set_geometry("geometry")
			df_dams.set_crs(epsg=4326, inplace=True)
			df_dams.to_file(r'%s\dams.shp' % (Basindir))
			df_dams.plot(ax=ax,marker='o',color='red')

	gdf_OCEAN = gpd.GeoDataFrame([[0]],columns=['idx'])
	gdf_OCEAN['geometry'] = list(cfeature.OCEAN.geometries())[0]
	gdf_OCEAN.set_crs(epsg=4326, inplace=True)
	gdf_OCEAN.to_file(r'%s\OCEAN.shp' % (Basindir))
	# -----------output stream and basin-----------
	gdf_Basin.set_crs(epsg=4326,inplace=True)
	gdf_Basin.to_file(r'%s\B%s.shp'%(Basindir, Basin_level))
	if gdf_Basin0 is not None:
		gdf_Basin0.set_crs(epsg=4326, inplace=True)
		gdf_Basin0.to_file(r'%s\B%s.shp' % (Basindir, Basin_level-1))
	if gdf_Basin_1 is not None:
		gdf_Basin_1.set_crs(epsg=4326, inplace=True)
		gdf_Basin_1.to_file(r'%s\B%s.shp' % (Basindir, Basin_level-2))
	if gdf_R1 is not None:
		gdf_R1.set_crs(epsg=4326,inplace=True)
		gdf_R1.to_file(r'%s\R%s.shp'%(Basindir,Stream_level))
	if gdf_R2 is not None:
		gdf_R2.set_crs(epsg=4326,inplace=True)
		gdf_R2.to_file(r'%s\R%s.shp' % (Basindir, Stream_level+1))
	if gdf_R3 is not None:
		gdf_R3 = gdf_R3.set_geometry("geometry")
		gdf_R3.set_crs(epsg=4326, inplace=True)
		gdf_R3.to_file(r'%s\R%s.shp' % (Basindir, Stream_level+2))
	if gdf_R0 is not None:
		gdf_R0.set_crs(epsg=4326,inplace=True)
		gdf_R0.to_file(r'%s\R%s.shp' % (Basindir, Stream_level-1))
	# -----------draw studied pnt-----------
	if gdf_pos is not None:
		gdf = gdf_pos
		gdf['Lon'] = [list(zip(*pnt.coords.xy))[0][0] for pnt in gdf['geometry']]
		gdf['Lat'] = [list(zip(*pnt.coords.xy))[0][1] for pnt in gdf['geometry']]
		for _, row in gdf.iterrows():
			pnt = (row['Lat'], row['Lon'])
			ax.scatter(pnt[1], pnt[0], marker='d', color='r', ec='r', lw=1, alpha=1, label='line1', s=10, zorder=8)

	if show:
		plt.savefig(os.path.join(Basindir, r'Hydromap.png'), format='png', dpi=600)
		plt.show()
	return extent

def dn_basin(Basin_level,ROI,Basindir=r'.'):
	WWF = ee.FeatureCollection("WWF/HydroATLAS/v1/Basins/level0%s" % Basin_level)
	Basin = WWF.filterBounds(ROI)
	gdf_Basin = geemap.ee_to_gdf(Basin)
	gdf_Basin.set_crs(epsg=4326,inplace=True)
	gdf_Basin.to_file(r'%s\B%s.shp'%(Basindir, Basin_level))


def Studied_reach( Regiondir, Basin_level=4, Stream_level=None, Reaches=None, draw_basin=True, ax=None ):
	from ...misc import load,projTrans
	Bankdir = os.path.join(Regiondir,r'Reaches\_Bank')
	shpfiles = [i for i in os.listdir(Bankdir) if os.path.splitext(i)[-1] == '.shp']
	years, reaches, areas, L1, L2, Lat, Lon, geometry = [], [], [], [], [], [], [], []
	for shpfile_ in shpfiles:
		shpfile = os.path.join(Bankdir, shpfile_)
		prjfile = os.path.join(shpfile.replace('.shp', '.prj'))
		proj = load(prjfile)

		pline1, pline2 = load(shpfile)
		years.append(os.path.splitext(shpfile_)[0][-4:])
		reaches.append(os.path.splitext(shpfile_)[0][:-5])
		L1.append(LineString(pline1).length)
		L2.append(LineString(pline2).length)
		Poly = Polygon(pline1 + pline2[::-1])
		areas.append(Poly.area)

		# inputDataSource = driver.Open(shpfile, 0)
		# inputLayer = inputDataSource.GetLayer(0)
		# shp_srs = inputLayer.GetSpatialRef()
		projTrans(shpfile, r'temp.shp', 4326)
		pline1_, pline2_ = load(r'temp.shp')
		Poly = Polygon(pline1_ + pline2_[::-1])
		lat, lon = list(zip(*Poly.centroid.coords.xy))[0]
		Lat.append(lat)
		Lon.append(lon)
		geometry.append( Point([lon,lat]) )
	df0 = pd.DataFrame({'Reach': reaches, 'area': areas, 'L1': L1, 'L2': L2, 'Lat': Lat, 'Lon': Lon}, columns=['Reach', 'area', 'L1', 'L2', 'Lat', 'Lon'])
	df0['geometry'] = geometry
	gdf = gpd.GeoDataFrame(df0)
	gdf = gdf.set_crs( epsg=4326, inplace=True, allow_override=False )
	gdf.to_file(Regiondir+r'\Reaches\Reaches_pos.shp')

	if ax is None:
		fig = plt.figure(figsize=(5.5, 4))
		ax = fig.add_axes([0.08, 0, 0.85, 0.5], projection=ccrs.PlateCarree())
		draw=True
	else:draw=False

	cmap = cmocean.cm.curl
	# cmap = pyris.get_cmap_from_img( r'C:\Users\Administrator\Python311\Lib\site-packages\pyris\img\cmap\dem1.png', start_from='t', N=None, buffer=12 )
	multipnts = [list(zip(*i.centroid.coords.xy))[0] for i in gdf['geometry']]
	# pyris.draw_dem(ax,file_path,elevation_thresh=2000, cmap=cmap,unit='km',vmin=-2e3, vmax=6000)
	ROI = ee.Geometry.MultiPoint(multipnts)

	Hydro_map( ax, ROI, Basin_level=4, Stream_level=Stream_level, DEM=True, res=2e3, shade_lim=[-60, 1.5e3], gdf_pos=gdf, draw_basin=draw_basin )
	# ax.set_xlim([82,96])
	# ax.set_ylim(extent[-2:])
	plt.savefig(os.path.join(Regiondir, r'Reaches\Studied_reaches.png'), format='png', dpi=600)
	if draw:
		plt.show()

def Basin_within_Basin( regiondir, ROI, Basin_level=4, Major_basin=None, overwrite=False ):
	if not os.path.isdir( regiondir ): os.mkdir( regiondir )
	if Major_basin is None:
		Major_basin = ee.FeatureCollection("WWF/HydroATLAS/v1/Basins/level0%s" % Basin_level).filterBounds(ROI)
	Basins = ee.FeatureCollection("WWF/HydroATLAS/v1/Basins/level0%s" % (Basin_level + 1)).filterBounds(Major_basin.geometry().buffer(-1e4))

	# gdf_Basins = geemap.ee_to_gdf(Basins)
	# gdf_Basins.plot()
	# plt.show()

	for idx,Basin in enumerate(Basins.toList(Basins.size().getInfo()).getInfo()):
		Basin_centroid = ee.Feature(Basin).centroid()
		Basinleveldir = r'%s\Basinlevel%s'%(regiondir,Basin_level)
		if not os.path.isdir(Basinleveldir): os.mkdir(Basinleveldir)
		Basindir = r'%s\Basin_%s'%(Basinleveldir,idx)
		if not os.path.isdir(Basindir): os.mkdir(Basindir)
		pngfile1 = r'%s\B%s_%s_Global.png' % (regiondir,Basin_level,idx)
		pngfile1_ = r'%s\B%s_%s_Global.png' % (Basindir, Basin_level, idx)
		pngfile2 = r'%s\B%s_%s_Local.png' % (regiondir,Basin_level,idx)
		pngfile2_ = r'%s\B%s_%s_Local.png' % (Basindir, Basin_level, idx)
		if not overwrite:
			if os.path.isfile(pngfile1) and os.path.isfile(pngfile2):continue

		fig = plt.figure(figsize=(5.5,4))
		ax = fig.add_axes([0.08, 0.2, 0.85, 0.8], projection=ccrs.PlateCarree())
		extent = Hydro_map( ax, ROI, Basin_level=Basin_level,Stream_level=None,DEM=False,res=None,shade_lim=[-60,1.5e3], stream_n=3, draw_basin=True, vis_param={'basin_color':'gray','R2_linestyle':'--','edgecolor':'k'}, draw_outer_stream=True, Basin=Major_basin, Basindir=Basindir )

		# ax.scatter( [pnt[0]], [pnt[1]] )
		Hydro_map( ax, Basin_centroid.geometry(), Basin_level=Basin_level+1, Stream_level=None, DEM=False, res=None, shade_lim=[-60, 1.5e3], stream_n=2, extent=extent, draw_scaler=False, draw_N=False, vis_param={'basin_color':'#2b2d30','R2_linestyle':'-'}, draw_outer_stream=False, Basindir=Basindir )

		plt.savefig( pngfile1, dpi=600,facecolor='w',edgecolor='w',orientation='portrait',format=None,transparent=False,bbox_inches=None,pad_inches=0.1,metadata=None )
		plt.savefig(pngfile1_, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', format=None,		transparent=False, bbox_inches=None,pad_inches=0.1, metadata=None)
		plt.show()

		fig = plt.figure(figsize=(5.5,4))
		ax = fig.add_axes([0.08, 0.15, 0.8, 0.8],projection=ccrs.PlateCarree())
		Hydro_map( ax, Basin_centroid.geometry(), Basin_level=Basin_level+1,Stream_level=None,DEM=True,res=None,shade_lim=[-60,1.5e3], stream_n=3, Basindir=Basindir )
		# plt.savefig( pngfile2, dpi=600,facecolor='w',edgecolor='w',orientation='portrait',format=None,transparent=False,bbox_inches=None,pad_inches=0.1,metadata=None )
		plt.savefig(pngfile2_, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
		plt.show()

"""
@Name: basic.py.py
@Auth: HeYucong (hyc0424@whu.edu.cn, 18188425393)
@Date: 2024/10/26-18:33
@Link: https://blog.csdn.net/awdwd233333
@Desc: 
@Ver : 0.0.0
"""
import os, matplotlib, rasterio
import numpy as np
import pandas as pd
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

plt.rcParams['axes.unicode_minus']=False#显示负号
plt.rcParams["font.size"]=10#全局字号,针对英文
plt.rcParams["mathtext.fontset"]='stix'#设置$$之间是字体与中文不一样,为新罗马,这样就可以中英混用
plt.rcParams['font.sans-serif'] = ['Times New Roman']#单独使用英文时为新罗马字体
def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3, units='km', m_per_unit=1000, fontsize=12):
	from .pntsTrans import pntsTrans
	"""

	http://stackoverflow.com/a/35705477/1072212
	ax is the axes to draw the scalebar on.
	proj is the projection the axes are in
	location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
	length is the length of the scalebar in km.
	linewidth is the thickness of the scalebar.
	units is the name of the unit
	m_per_unit is the number of meters in a unit
	"""
	# find lat/lon center to find best UTM zone
	x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
	# Projection in metres
	EPSG = 32600 + int((x0 + x1) / 2 // 6 + 31)
	utm = ccrs.UTM(int(floor((x0 + x1) / 2 / 6) + 31))

	# Get the extent of the plotted area in coordinates in metres
	x0, x1, y0, y1 = ax.get_extent(utm)
	# Turn the specified scalebar location into coordinates in metres
	sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
	# Generate the x coordinate for the ends of the scalebar
	bar_xs = [sbcx - length * m_per_unit / 2, sbcx + length * m_per_unit / 2]
	# buffer for scalebar
	buffer = [patheffects.withStroke(linewidth=linewidth + 2, foreground="w")]
	# Plot the scalebar with buffer
	line, = ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k', linewidth=linewidth, path_effects=buffer)
	# buffer for text
	buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
	# Plot the scalebar label
	t0 = ax.text(sbcx, sbcy + (y1 - y0) * 0.02, str(length) + ' ' + units, transform=utm, horizontalalignment='center',
	             fontsize=fontsize, verticalalignment='bottom', path_effects=buffer, zorder=10)

	left = x0 + (x1 - x0) * 0.05
	bb_line = find_bbox_of_obj(ax, line)
	bb_t = find_bbox_of_obj(ax, t0)
	l = pntsTrans([[(bar_xs[0], sbcy), (bar_xs[1], sbcy)]], EPSG, 4326)[0]

	# Plot the N arrow
	# l = pntsTrans([[(bb_line.x1,bb_t.y1)]], 4326, EPSG)[0]
	# sbcx_N, sbcy_N = l[0][0],l[0][1]
	# t1 = ax.text( sbcx_N, sbcy_N, u'\u25B2\nN', transform=utm,
	# 	horizontalalignment='right', verticalalignment='bottom',
	# 	path_effects=buffer, zorder=2, fontsize=fontsize )
	# bb_t1 = find_bbox_of_obj(ax, t1)

	t_x0, t_x1, t_y0, t_y1 = min(bb_line.x0, bb_t.x0), max(bb_line.x1, bb_t.x1), bb_line.y0, bb_t.y1
	# Plot the scalebar without buffer, in case covered by text buffer
	ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k', linewidth=linewidth, zorder=3)
	return [t_x0, t_x1, t_y0, t_y1]


def N_arrow( ax, proj, location=(0.5, 0.05), fontsize=12 ):
	buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
	x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
	sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
	t1 = ax.text(sbcx, sbcy, u'\u25B2\nN', horizontalalignment='center', verticalalignment='bottom',
	             path_effects=buffer, zorder=10, fontsize=fontsize)
	bb_t1 = find_bbox_of_obj(ax, t1)
	t_x0, t_x1, t_y0, t_y1 = bb_t1.x0, bb_t1.x1, bb_t1.y0, bb_t1.y1
	return [t_x0, t_x1, t_y0, t_y1]


def find_bbox_of_obj(ax, obj):
	transf = ax.transData.inverted()
	bb = obj.get_window_extent(renderer=find_renderer(ax.get_figure()))
	bb_datacoords = bb.transformed(transf)
	return bb_datacoords


def find_renderer(fig):
	if hasattr(fig.canvas, "get_renderer"):
		# Some backends, such as TkAgg, have the get_renderer method, which
		# makes this easy.
		renderer = fig.canvas.get_renderer()
	else:
		# Other backends do not have the get_renderer method, so we have a work
		# around to find the renderer.  Print the figure to a temporary file
		# object, and then grab the renderer that was used.
		# (I stole this trick from the matplotlib backend_bases.py
		# print_figure() method.)
		import io
		fig.canvas.print_pdf(io.BytesIO())
		renderer = fig._cachedRenderer
	return renderer


def find_position(ax, geom, shape, y_bias=None, x_bias=None):
	'''
	Find the position of the obj with shape, that is not intersect with geom
	:param ax:
	:param geom: shapely geometry object
	:param shape: shape coords
	:param y_bias:
	:param x_bias:
	:return:
	'''
	if x_bias is None: x_bias = shape[0] * 0.2
	if y_bias is None: y_bias = shape[1] * 0.2
	import shapely
	ax_rect = Polygon([(ax.get_xlim()[0], ax.get_ylim()[0]), (ax.get_xlim()[0], ax.get_ylim()[1]),
	                   (ax.get_xlim()[1], ax.get_ylim()[1]), (ax.get_xlim()[1], ax.get_ylim()[0])])
	find_pos = False
	for y in np.linspace(ax.get_ylim()[0], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]), 30):
		for x in np.linspace(ax.get_xlim()[0], ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2,
		                     30).tolist() + np.linspace(ax.get_xlim()[1], ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2,30).tolist():
			pos = (x, y)
			x0 = pos[0] - shape[0] / 2 - x_bias
			y0 = pos[1] - y_bias
			x1 = pos[0] + shape[0] / 2 + x_bias
			y1 = pos[1] + shape[1] + 2 * y_bias
			Rect_poly = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
			# print(Rect_poly)
			if not shapely.intersects(Rect_poly, geom) and (shapely.covers(ax_rect, Rect_poly)):
				find_pos = True
			if find_pos == True: break
		if find_pos == True: break

	if find_pos == False:
		print('change to', ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02)
		ax.set_ylim([ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, ax.get_ylim()[1]])
		pos, box = find_position(ax, geom, shape)
	else:
		gpd.GeoSeries(Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])).plot(color='w', alpha=0, ax=ax)
		pos = [pos[0], pos[1] + y_bias]
		box = [x0, x1, y0, y1]
	return pos, box


def all_geom_intersects(gdf, buffer=0.1):
	'''
	check if there is any geom in gdf that is independent from others, i.e., isolated geom in gdf exist
	:param gdf:
	:param buffer:
	:return:
	'''
	for i in range(len(gdf.geometry) - 1):
		# test
		# fig = plt.figure(figsize=(5.5,4))
		# gdf.plot()
		#
		# l = gdf.geometry[i]
		# x, y = list(zip(*list(zip(*l.coords.xy))))
		# line, = plt.plot(x, y, 'r,-', marker=',', c='r', ms=1, ls='-', lw=1, markerfacecolor='none', alpha=1, label='line1')
		#
		# # for t in range(len(gdf.geometry)):
		# # 	l = gdf.geometry[t].buffer(buffer)
		# # 	x, y = list(zip(*list(zip(*l.exterior.coords.xy))))
		# # 	line, = plt.plot(x, y, 'b,-', marker=',', c='b', ms=0.03, ls='-', lw=0.2, markerfacecolor='b', alpha=1,label='line1')
		# plt.show()
		flag = [gdf.geometry[i].buffer(buffer).intersects(gdf.geometry[j].buffer(buffer)) for j in range(len(gdf.geometry))]
		# if np.array(flag).astype('int').sum() == 0:  # no any other geom covers geom i
		# 	return False
	return True


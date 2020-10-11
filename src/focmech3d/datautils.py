#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>

import os, sys
import pandas as pd

def readingfile(filename, path = None):
	if path == None:
		path = os.getcwd()
	f=os.path.join(path, filename)
	if not os.path.isfile(f):
		sys.exit('File(s) missing:'+f)
	return f

def createpath(directory):
	if not os.path.isdir(directory):
		os.mkdir(directory)
	return directory

def load_data(filename, **kwargs):
	filetype = kwargs.pop('filetype', None)

	try:
		if filetype == 'excel':
			df = pd.read_excel(filename, **kwargs)
		else:
			df = pd.read_csv(filename, **kwargs)
	except ValueError:
		try:
			df = pd.read_csv(filename, sep = None, engine = 'python', **kwargs)
		except ValueError:
			raise Exception('Could not find the appropriate number of columns. Try specifying the delimiter with the keyword sep.')

	#reorder columns
	if 'usecols' in kwargs:
		columns = sorted(kwargs['usecols']) #this is the order that Pandas put the columns in
		map_usecol_to_order = {col: num for num, col in enumerate(columns)} #which position the column number from kwargs['usecols'] is in
		new_ordering = [map_usecol_to_order[col] for col in kwargs['usecols']]
		colnames = [df.columns[i] for i in new_ordering]
		df = df[colnames]
		kwargs.pop('usecols')
	return df, kwargs
	



def make_test_data():
	center = [0, 0, 0]
	radius = 1
	data = []
	for strike in range(0, 360, 30):
		for dip in range(-180, 180, 30):
			for slip in range(-180, 180, 30):
				data.append([radius, center, [strike, dip, slip]])

	return data
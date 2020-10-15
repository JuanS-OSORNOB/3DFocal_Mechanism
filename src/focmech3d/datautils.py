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

def load_data(filename, usecols, filetype = None, sep = None, sheet_name = 0):
	try:
		if filetype == 'excel':
			df = pd.read_excel(filename, usecols = usecols, sheet_name = sheet_name)
		elif sep is None:
			df = pd.read_csv(filename, sep = None, engine = 'python', usecols = usecols)
		else:
			df = pd.read_csv(filename, usecols = usecols, sep = sep)
	except ValueError:
		try:
			df = pd.read_csv(filename, sep = None, engine = 'python', usecols = usecols)
		except ValueError:
			raise Exception('Could not find the appropriate number of columns. Try specifying the delimiter with the keyword sep.')

	#reorder columns
	columns = sorted(usecols) #this is the order that Pandas put the columns in
	if columns != usecols:
		map_usecol_to_order = {col: num for num, col in enumerate(columns)} #which position the column number from kwargs['usecols'] is in
		new_ordering = [map_usecol_to_order[col] for col in usecols]
		colnames = [df.columns[i] for i in new_ordering]
		df = df[colnames]
	return df
	



def make_test_data():
	center = [0, 0, 0]
	radius = 1
	data = []
	for strike in range(0, 360, 30):
		for dip in range(-180, 180, 30):
			for slip in range(-180, 180, 30):
				data.append([radius, center, [strike, dip, slip]])

	return data
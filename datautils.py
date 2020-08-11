import os, sys

def readingfile(path, file):
	file=os.path.join(path, file)
	if not os.path.isfile(file):
		sys.exit('File(s) missing:'+file)
	return path, file

def createpath(directory):
	if not os.path.isdir(directory):
		os.mkdir(directory)
	return directory

def parse_file(filename, header):
	#parse tab-delimited file into floats
	data = []
	with open(filename) as f:
		if header: #skip first line
			headers = f.readline()
		for line in f.readlines():
			line = line.strip().split('\t')
			line = [float(x) for x in line]
			x, y, z = line[1:4]
			
			entry = [line[0], [x, y, -1*z], line[4:7]]
			data.append(entry)
	return data


def make_test_data():
	center = [0, 0, 0]
	radius = 1
	data = []
	for strike in range(0, 360, 30):
		for dip in range(-180, 180, 30):
			for slip in range(-180, 180, 30):
				data.append([radius, center, [strike, dip, slip]])

	return data
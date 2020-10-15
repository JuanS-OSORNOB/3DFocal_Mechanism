#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>

from focmech3d.datautils import load_data
from focmech3d.focal_mechanism import FocalMechanism, Event

def load_events(filename, projection = 'equirectangular', rad_function = lambda x: x, 
                invert_z = True, filetype = None, usecols = None, sheet_name = 0, colnames = None):
    '''Creates an iterable of Event objects by loading event data and creating an Event object from each line.
    The parameters longitude, latitude, altitude, magnitude are loaded from the given file,
    in that order. By default, the first four columns of the file are used. If you want 
    to use a different set of columns (or the same set but in a different order), use the 
    keyword 'usecols'. For example, if your data is in the order magnitude, latitude, 
    longitude, altitude, you should use usecols = [2, 1, 3, 0]. If you choose to use additional
    columns, the extra data will be stored in the attribute Event.other_params.
    Keyword arguments:
        filetype: 'excel' or None. If 'excel', uses pandas.read_excel; otherwise, uses pandas.read_csv.
        projection: 'equirectangular'. (This exists for future expansion; equirectangular projection is the only option at this time)
        usecols: List of integers (default range(4)). Indicates which columns to use, and in what order.
    This function uses pandas.read_csv or pandas.read_excel to read the file. Please see the documentation 
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html and
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html for keyword arguments. '''

    if 'usecols' is None:
        usecols = range(4)
    data = load_data(filename, usecols, filetype = filetype, sheet_name = sheet_name)
    events = generate_events_from_dataframe(data, colnames = colnames, projection = projection, rad_function = rad_function,
                                                        invert_z = invert_z)
    return events

def load_fms(filename, projection = 'equirectangular', in_degrees = True, rad_function = lambda x: x, invert_z = True, 
            filetype = None, usecols = None, sheet_name = 0, colnames = None):
    '''Creates an iterable of FocalMechanism objects by loading focal mechanism data and creating an FocalMechanism object from each line.
    The parameters longitude, latitude, altitude, magnitude, strike, dip, rake are loaded from the given file,
    in that order. By default, the first seven columns of the file are used. If you want 
    to use a different set of columns (or the same set but in a different order), use the 
    keyword 'usecols'. For example, if your data is in the order magnitude, latitude, 
    longitude, altitude, strike, dip, rake, you should use usecols = [2, 1, 3, 0, 4, 5, 6]. If you choose to use additional
    columns, the extra data will be stored in the attribute FocalMechanism.other_params.
    Keyword arguments:
        filetype: 'excel' or None. If 'excel', uses pandas.read_excel; otherwise, uses pandas.read_csv.
        projection: 'equirectangular'. (This exists for future expansion; equirectangular projection is the only option at this time)
        in_degrees: True or False. If True, indicates that the strike, dip, and rake are in degrees. If False, they are in radians.
        usecols: List of integers (default range(7)). Indicates which columns to use, and in what order.
    This function uses pandas.read_csv or pandas.read_excel to read the file. Please see the documentation 
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html and
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html for keyword arguments. '''

    if usecols is None:
        usecols = range(7)
    data = load_data(filename, usecols, filetype = filetype, sheet_name = sheet_name)
    focal_mechanism_list = generate_fms_from_dataframe(data, colnames = colnames, projection = projection, in_degrees = in_degrees, rad_function = rad_function,
                                                        invert_z = invert_z)
    return focal_mechanism_list

def generate_fms_from_list(parameter_list, colnames = [], projection = 'equirectangular', 
                            in_degrees = True, rad_function = lambda x: x, invert_z = True):
    '''Use if you have a list of parameters that you want to use to generate the focal mechanisms. '''
    fms = []
    for parameters in parameter_list:
        fm = FocalMechanism(*parameters, colnames = colnames, projection = projection, in_degrees = in_degrees, 
                            rad_function = rad_function, invert_z = invert_z)
        fms.append(fm)
    return fms

def generate_fms_from_dataframe(df, colnames = None, projection = 'equirectangular', 
                                in_degrees = True, rad_function = lambda x: x, invert_z = True):
    '''Use if you want to generate focal mechanisms from a DataFrame object.'''
    if colnames is None:
            colnames = df.columns.tolist()[7:]
    data = zip(*[df[col] for col in df])
    fms = generate_fms_from_list(data, colnames = colnames, projection = projection, in_degrees = in_degrees, rad_function = rad_function,
    invert_z = invert_z)
    return fms

def generate_events_from_list(parameter_list, colnames = [], projection = 'equirectangular', 
                            rad_function = lambda x: x, invert_z = True):
    '''Use if you have a list of parameters that you want to use to generate the events. '''
    events = []
    for parameters in parameter_list:
        event = Event(*parameters, colnames = colnames, projection = projection, 
                            rad_function = rad_function, invert_z = invert_z)
        events.append(event)
    return events

def generate_events_from_dataframe(df, colnames = None, projection = 'equirectangular', 
                                rad_function = lambda x: x, invert_z = True):
    '''Use if you want to generate events from a DataFrame object.'''
    if colnames is None:
            colnames = df.columns.tolist()[4:]
    data = zip(*[df[col] for col in df])
    events = generate_events_from_list(data, colnames = colnames, projection = projection, rad_function = rad_function, invert_z = invert_z)
    return events
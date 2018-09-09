import shapefile
import math
import utm
import csv


# Set Coordinate System
#   0 = Use Latitude and Longitude
#   1 = Use UTM
use_utm = 0


def get_quadrat():
    #   Create 2 meter bounding coordinates for a quadrat.
    if use_utm:
        center = utm.from_latlon(lat, lon)
        x_offset = 1
        y_offset = 1
        upright = [center[0] + y_offset, center[1] + x_offset]
        downright = [center[0] + y_offset, center[1] - x_offset]
        upleft = [center[0] - y_offset, center[1] + x_offset]
        downleft = [center[0] - y_offset, center[1] - x_offset]
    else:
        center = [lat, lon]
        x_offset = 1 / (111111 * math.cos(lat))
        y_offset = 1 / 111111
        upright = [center[1] + x_offset, center[0] + y_offset]
        downright = [center[1] + x_offset, center[0] - y_offset]
        upleft = [center[1] - x_offset, center[0] + y_offset]
        downleft = [center[1] - x_offset, center[0] - y_offset]
    return upleft, upright, downright, downleft


def make_poly():
    #   makes a square with (lat, lon) as center coordinate and loops num times
    w = shapefile.Writer(shapefile.POLYGON)
    seen.append(name)
    upleft, upright, downright, downleft = get_quadrat()
    w.poly(parts=[[[upright[0], upright[1]], [downright[0],
                   downright[1]], [downleft[0], downleft[1]],
                  [upleft[0], upleft[1]]]])
    w.field('Quadrat', 'C')
    w.field('Date', 'C')
    w.field('Latitude', 'N', decimal=10)
    w.field('Longitude', 'N', decimal=10)
    w.field('Max_Height', 'N', decimal=2)
    w.record(Quadrat=name, Date=date, Latitude=lat, Longitude=lon,
             Max_Height=height)
    w.save('shapefiles/polygon')
    w = None


def edit_poly():
    # checks for duplicates
    for names in seen:
        if names == name:
            return
    seen.append(name)
    # adds more shapes to current shapefile
    e = shapefile.Editor(shapefile="shapefiles/polygon.shp")
    upleft, upright, downright, downleft = get_quadrat()
    e.poly(parts=[[[upright[0], upright[1]], [downright[0],
                   downright[1]], [downleft[0], downleft[1]],
                  [upleft[0], upleft[1]]]])
    e.record(Quadrat=name, Date=date, Latitude=lat, Longitude=lon,
             Max_Height=height)
    # e.record(Latitude=lat, Longitude=lon)
    e.save('shapefiles/polygon')
    e = None


file_name = input('Import csv file: ')
num = 0
with open(file_name, mode='r') as f:
        reader = csv.DictReader(f)
        # stores already written coordinates to skip duplicate quadrats
        seen = []
        for row in reader:
            # checks if row is blank and skips it
            if not (row['Latitude'] and row['Longitude']):
                num += 1
                continue
            # make initial polygon and file
            num += 1
            name = row['Polygon']
            date = row['Trip']
            height = row['Max Canopy Height']
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            if num == 1:
                make_poly()
            else:
                edit_poly()

import shapefile
import utm
import csv


def makePoly(lat, lon, name, date, loc, seen):
    #   makes a square with (lat, lon) as center coordinate
    w = shapefile.Writer(shapefile.POLYGON)
    seen.append(name)
    utm_center = utm.from_latlon(lat, lon)
    utm_upright = [utm_center[0] + 1, utm_center[1] + 1]
    utm_downright = [utm_center[0] + 1, utm_center[1] - 1]
    utm_upleft = [utm_center[0] - 1, utm_center[1] + 1]
    utm_downleft = [utm_center[0] - 1, utm_center[1] - 1]
    w.poly(parts=[[[utm_upright[0], utm_upright[1]], [utm_downright[0],
                   utm_downright[1]], [utm_downleft[0], utm_downleft[1]],
                  [utm_upleft[0], utm_upleft[1]]]])
    w.field('Quadrat', 'C')
    w.field('Location', 'C')
    w.field('Date', 'C')
    w.field('Latitude', 'N', decimal=10)
    w.field('Longitude', 'N', decimal=10)
    w.record(Quadrat=name, Date=date, Location=loc, Latitude=lat,
             Longitude=lon)
    # w.record(Latitude=lat, Longitude=lon)
    w.save('shapefiles/polygon')


def editPoly(lat, lon, name, date, loc, seen):
    # checks for duplicates
    for names in seen:
        if names == name:
            return
    seen.append(name)
    # adds more shapes to current shapefile
    e = shapefile.Editor(shapefile="shapefiles/polygon.shp")
    utm_center = utm.from_latlon(lat, lon)
    utm_upright = [utm_center[0] + 1, utm_center[1] + 1]
    utm_downright = [utm_center[0] + 1, utm_center[1] - 1]
    utm_upleft = [utm_center[0] - 1, utm_center[1] + 1]
    utm_downleft = [utm_center[0] - 1, utm_center[1] - 1]
    e.poly(parts=[[[utm_upright[0], utm_upright[1]], [utm_downright[0],
                   utm_downright[1]], [utm_downleft[0], utm_downleft[1]],
                  [utm_upleft[0], utm_upleft[1]]]])
    e.record(Quadrat=name, Date=date, Location=loc, Latitude=lat,
             Longitude=lon)
    # e.record(Latitude=lat, Longitude=lon)
    e.save('shapefiles/polygon')


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
            loc = row['Location']
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            if num == 1:
                makePoly(lat, lon, name, date, loc, seen)
            else:
                editPoly(lat, lon, name, date, loc, seen)

import shapefile
import utm
import sys
w = shapefile.Writer(shapefile.POLYGON)
lat = float(sys.argv[1])
long = float(sys.argv[2])
utm_center = utm.from_latlon(lat, long)
utm_upright = [utm_center[0] + 1, utm_center[1] + 1]
utm_downright = [utm_center[0] + 1, utm_center[1] - 1]
utm_upleft = [utm_center[0] - 1, utm_center[1] + 1]
utm_downleft = [utm_center[0] - 1, utm_center[1] - 1]
w.poly(parts=[[[utm_upright[0], utm_upright[1]], [utm_downright[0],
               utm_downright[1]], [utm_downleft[0], utm_downleft[1]],
              [utm_upleft[0], utm_upleft[1]]]])
w.field('FIRST_FLD', 'C', '40')
w.field('SECOND_FLD', 'C', '40')
w.record('First', 'Square')
w.save('1square/polygon')

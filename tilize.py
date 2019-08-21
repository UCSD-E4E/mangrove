import gdal
import os
import argparse
import tqdm
import math

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help='input file')
    parser.add_argument('--outdir', help='output directory')
    parser.add_argument('-s', '--shape', type=int, default=128, help='tile side length in px')
    args = parser.parse_args()

    infile = os.path.abspath(args.infile)
    outdir = os.path.abspath(args.outdir)
    side = args.shape

    ds = gdal.Open(infile)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    os.makedirs(outdir, exist_ok=True)
    pbar = tqdm.tqdm(total=math.ceil(xsize/side)*math.ceil(ysize/side))
    for i in range(0, xsize, side):
        for j in range(0, ysize, side):
            # command = f"gdal_translate -of GTIFF -srcwin {i}, {j}, {side}, {side} {infile} {outdir}/tile_{i}_{j}.tif"
            # os.system(command)
            opts = gdal.TranslateOptions(format='GTiff', srcWin=[i, j, side, side], creationOptions=['COMPRESS=DEFLATE'])
            gdal.Translate(os.path.join(outdir, f"tile_{i}_{j}.tif"), ds, options=opts)
            pbar.update(1)

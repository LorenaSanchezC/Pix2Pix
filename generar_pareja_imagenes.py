from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdalconst
import mapscript
from csv import reader
import getopt, sys
import os, shutil
import time, glob, math
from random import random
import urllib.request
import urllib.error
import numpy as np

#import socket
#socket.setdefaulttimeout(time = 120) # 120 seconds


def download_image(url, fullname):
    try:
        urllib.request.urlretrieve(url,fullname)
        return True
    except urllib.error.URLError as e:
        time.sleep(20 + 10 *random())
        try:
            urllib.request.urlretrieve(url,fullname)
            return True            
        except urllib.error.URLError as e:
            time.sleep(30 + 20 *random())
            try:
                urllib.request.urlretrieve(url,fullname)
                return True                
            except urllib.error.URLError as e:    
                time.sleep(60 + 20 *random())
                try:
                    urllib.request.urlretrieve(url,fullname)
                    return True                    
                except urllib.error.URLError as e:
                    print("Failed 4 times to download '{}'. '{}'".format(url, e.reason))
                    return False

def convert_png_gtiff(src_filename, dst_filename, xmin, ymax, xsize, ysize, CRS ):
    src_ds = gdal.Open(src_filename)
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.CreateCopy(dst_filename, src_ds, 1)
    gt = [xmin, xsize, 0, ymax, 0, ysize]
    dst_ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(CRS))
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds = None
    src_ds = None


gdal_env = os.environ.copy()

# modify and add variables
#gdal_env["GDAL_DATA"]
#os.environ["PROJ_LIB"] 

url_ini_pattern = "http://ree.geoide.upm.es/cgi-bin/pnoa-ma?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=OI.OrthoimageCoverage&STYLES=DEFAULT&FORMAT=image%2Fpng&BBOX={}&CRS=EPSG%3A{}&WIDTH=256&HEIGHT=256"


# Establece nombre de directorio de salida de las imágenes
imgsFolder = 'ejes_img'

# Leer parámetros de la llamada
argumentList = sys.argv[1:] 
# Options 
options = "hf:e:b:m:d:c:"
# Long options 
long_options = ["Help", "csv_centros", "map_file=", "Img_Directory=", "CRS="] 

csvCenter = ''
mapFile = ''
CRS = ''
img_width, img_height = 512, 512


    
if __name__ == '__main__':
    try: 
        # Parsing argument 
        arguments, values = getopt.getopt(argumentList, options, long_options) 
          
        # checking each argument 
        for currentArgument, currentValue in arguments: 
      
            if currentArgument in ("-h", "--Help"): 
                print ("Diplaying Help")
                print ("-f Input image; -ns H5 CNN file name; -nc H5 Cla file name; -d Img_Directory; -c CRS")
            elif currentArgument in ("-f", "--csv_centros"): 
                print ("CSV centroides (% s)" % (currentValue))
                csvCenter = currentValue
            elif currentArgument in ("-m", "--map_file"): 
                print (("Mapfile (% s)") % (currentValue))
                mapFile = currentValue                              
            elif currentArgument in ("-c", "--CRS"): 
                print (("Img CRS (% s)") % (currentValue))
                CRS = currentValue                 

    except getopt.error as err: 
        # output error, and return with an error code 
        print (str(err))

    if not CRS or not mapFile or not csvCenter: 
        sys.exit("No hay suficientes parámetros: csvEjes, MapFile or CRS")


    imgPathroot = os.getcwd() + '/' + imgsFolder + '_' + str(img_width)
    if not os.path.isdir(imgPathroot):
        os.makedirs(imgPathroot)
    imgPatheje = os.getcwd() + '/' + imgsFolder + '_' + str(img_width) + '/eje'
    if not os.path.isdir(imgPatheje):
        os.makedirs(imgPatheje)
    imgPathbuff = os.getcwd() + '/' + imgsFolder + '_' + str(img_width) + '/buf'
    if not os.path.isdir(imgPathbuff):
        os.makedirs(imgPathbuff)
    
    # Patrón de las tiles de imágenes segmentados a generar    
    fN_pattern_ejes = imgPatheje + '/neje_{}.png'        
    fN_pattern_buff = imgPathbuff + '/neje_{}.png' 


    mapscript.msIO_installStdoutToBuffer()
    reqBuff = mapscript.OWSRequest()
    reqBuff.setParameter( 'SERVICE', 'WMS' )
    reqBuff.setParameter( 'VERSION', '1.1.1' )
    reqBuff.setParameter( 'REQUEST', 'GetMap' )
    reqBuff.setParameter('STYLES', ',,')
    reqBuff.setParameter('WIDTH', str(img_width))
    reqBuff.setParameter('HEIGHT', str(img_height))
    reqBuff.setParameter('SRS', 'EPSG:' + CRS)
    reqBuff.setParameter('FORMAT', 'image/png')
    reqBuff.setParameter('LAYERS','buff')

    reqEjes = mapscript.OWSRequest()
    reqEjes.setParameter( 'SERVICE', 'WMS' )
    reqEjes.setParameter( 'VERSION', '1.1.1' )
    reqEjes.setParameter( 'REQUEST', 'GetMap' )
    reqEjes.setParameter('STYLES', ',,')
    reqEjes.setParameter('WIDTH', str(img_width))
    reqEjes.setParameter('HEIGHT', str(img_height))
    reqEjes.setParameter('SRS', 'EPSG:' + CRS)
    reqEjes.setParameter('FORMAT', 'image/png')
    reqEjes.setParameter('LAYERS','ejes')

    map = mapscript.mapObj( os.getcwd() + '/' + mapFile )
    
    with open(csvCenter, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=',')
        next(csv_reader)
        cnt = 0
        for row in csv_reader:
            #print(row[0], row[1]) #, row[3])
            xc = float(row[1])
            yc = float(row[2])
            xmin= xc - 128
            xmax = xc + 128
            ymin = yc - 128
            ymax = yc + 128
            coords = str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax)
            fName_ejes = fN_pattern_ejes.format(row[0])
            fName_buff = fN_pattern_buff.format(row[0])


            try: 
                reqEjes.setParameter('BBOX', coords)
                status = map.OWSDispatch( reqEjes )            
                assert status == 0
                headers = mapscript.msIO_getAndStripStdoutBufferMimeHeaders()
                assert headers is not None
                assert 'Content-Type' in headers
                assert headers['Content-Type'] == 'image/png'

                result = mapscript.msIO_getStdoutBufferBytes()
                assert result is not None
                assert result[1:4] == b'PNG'

                with open(fName_ejes, "wb") as f:
                    f.write(result)
            except:
                print("Error en petición WMS Ejes", str(row[0]))
                    
            try:   
                reqBuff.setParameter('BBOX', coords)
                status = map.OWSDispatch( reqBuff )
            
                assert status == 0
                headers = mapscript.msIO_getAndStripStdoutBufferMimeHeaders()
                assert headers is not None
                assert 'Content-Type' in headers
                assert headers['Content-Type'] == 'image/png'

                result1 = mapscript.msIO_getStdoutBufferBytes()
                assert result1 is not None
                assert result1[1:4] == b'PNG'

                with open(fName_buff, "wb") as f1:
                    f1.write(result1)                
            except:
                print("Error en petición WMS Buffer ", str(row[0]) )
            cnt += 1
        print ("Number of features in %s: %d" % ( CRS , cnt ))


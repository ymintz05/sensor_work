#!/usr/bin/env python

#== SAMPLE ID ==#
# 0K7005W5
#===============#

from gdx import gdx
from vpython import *
gdx = gdx.gdx()

gdx.open(connection='usb')
gdx.select_sensors()
gdx.vp_vernier_canvas()    
b = box(size=0.1*vec(1,1,1), color=color.red)
gdx.start(period=250)
 
while gdx.vp_close_is_pressed() == False:
    while gdx.vp_collect_is_pressed() == True:       
        measurements = gdx.read()     
        sensor0_data =  measurements[0]  
        b.length = 0.1 * sensor0_data 
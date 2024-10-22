#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Dynamic Weather:

Connect to a CARLA Simulator instance and control the weather. Change Sun
position smoothly with time and generate storms occasionally.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import math

def set_weather(world, weather_params):
    weather = carla.WeatherParameters(
        cloudiness=weather_params['cloudiness'],
        precipitation=weather_params['precipitation'],
        precipitation_deposits=weather_params['precipitation_deposits'],
        sun_altitude_angle=weather_params['sun_altitude_angle'],
        fog_density=weather_params['fog_density'],
        fog_distance=weather_params['fog_distance']
    )
    world.set_weather(weather)


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-s', '--situation',
        metavar='SITUATION',
        default='clear-day',
        help='select the situation, possible values are clear-day(defualt), clear-night, rainy-night')
    args = argparser.parse_args()
    # Connect to CARLA server and get the 
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    if args.situation == 'clear-day':
        # Define weather parameters
        weather_params = {
            'cloudiness': 0,
            'precipitation': 0,
            'precipitation_deposits': 0, # 
            'sun_altitude_angle': 45.0, # negative value means night -90 midnight 
            'fog_density': 0, # 0 - 100
            'fog_distance': 0 # starting point of fog. 0 means infinate
        }
    elif args.situation == 'rainy-day':
        # Define weather parameters
        weather_params = {       
            'cloudiness': 100,
            'precipitation': 100,
            'precipitation_deposits': 100, # 
            'sun_altitude_angle': 45.0, # negative value means night -90 midnight 
            'fog_density': 75, # 0 - 100
            'fog_distance': 0 # starting point of fog. 0 means infinate
        }      
    elif args.situation == 'clear-night':
        # Define weather parameters
        weather_params = {       
            'cloudiness': 0,
            'precipitation': 0,
            'precipitation_deposits': 0, # 
            'sun_altitude_angle': -90.0, # negative value means night -90 midnight 
            'fog_density': 0, # 0 - 100
            'fog_distance': 0 # starting point of fog. 0 means infinate     
        }
    elif args.situation == 'rainy-night':
        # Define weather parameters
        weather_params = {       
            'cloudiness': 100,
            'precipitation': 100,
            'precipitation_deposits': 100, # 
            'sun_altitude_angle': -90.0, # negative value means night -90 midnight 
            'fog_density': 75, # 0 - 100
            'fog_distance': 0 # starting point of fog. 0 means infinate
        }  

    # Set the desired weather conditions
    set_weather(world, weather_params)



if __name__ == '__main__':

    main()

#!/usr/bin/env python

"""
An example of client-side bounding boxes with basic car controls.

Controls:
Welcome to CARLA for Getting Bounding Box Data.
Use WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    P            : autopilot mode
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import threading
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc
# import VehiclePIDController
import weakref
#from agents.navigation.basic_agent import BasicAgent
import random
import cv2
import time
import argparse
import textwrap
"""
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_TAB
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_p
    from pygame.locals import K_c
    from pygame.locals import K_l
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
"""
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 960
VIEW_HEIGHT = 540
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)
WBB_COLOR = (0, 0, 255)
vehicle_bbox_record_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
pedestrian_bbox_record_list = [False, False, False, False, False, False, False, False, False, False, False, False, False]
count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lock = threading.Lock()
rgb_info = np.zeros((540, 960, 3), dtype="i")
seg_info = np.zeros((540, 960, 3), dtype="i")

# Creates Directory
dir_rgb = 'custom_data/'
dir_seg = 'SegmentationImage/'
dir_pbbox = 'PedestrianBBox/'
dir_vbbox = 'VehicleBBox/'
if not os.path.exists(dir_rgb):
    os.makedirs(dir_rgb)
if not os.path.exists(dir_seg):
    os.makedirs(dir_seg)
if not os.path.exists(dir_pbbox):
    os.makedirs(dir_pbbox)
if not os.path.exists(dir_vbbox):
    os.makedirs(dir_vbbox)

# ==============================================================================
# -- PedestrianBoundingBoxes ---------------------------------------------------
# ==============================================================================

class PedestrianBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(pedestrians, camera):
        """
        Creates 3D bounding boxes based on carla Pedestrian list and camera.
        """

        bounding_boxes = [PedestrianBoundingBoxes.get_bounding_box(pedestrian, camera) for pedestrian in pedestrians]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, car_id):
        """
        Draws bounding boxes on pygame display.
        """
        ##bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        ##bb_surface.set_colorkey((0, 0, 0))
        
        if pedestrian_bbox_record_list[car_id] == True:
            f = open(f'PedestrianBBox/{car_id}/bbox{str(count_list[car_id])}', 'w')
            print(f'PedestrianBBox/{car_id}/bbox{str(count_list[car_id])}')
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            if pedestrian_bbox_record_list[car_id] == True:
                f.write(str(points)+"\n")
        
        if pedestrian_bbox_record_list[car_id] == True:
            f.close()
            pedestrian_bbox_record_list[car_id] = False

        ##display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(pedestrian, camera):
        """
        Returns 3D bounding box for a pedestrian based on camera view.
        """

        bb_cords = PedestrianBoundingBoxes._create_bb_points(pedestrian)
        cords_x_y_z = PedestrianBoundingBoxes._pedestrian_to_sensor(bb_cords, pedestrian, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(pedestrian):
        """
        Returns 3D bounding box for a pedestrian.
        """

        cords = np.zeros((8, 4))
        extent = pedestrian.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _pedestrian_to_sensor(cords, pedestrian, sensor):
        """
        Transforms coordinates of a pedestrian bounding box to sensor.
        """

        world_cord = PedestrianBoundingBoxes._pedestrian_to_world(cords, pedestrian)
        sensor_cord = PedestrianBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _pedestrian_to_world(cords, pedestrian):
        """
        Transforms coordinates of a pedestrian bounding box to world.
        """

        bb_transform = carla.Transform(pedestrian.bounding_box.location)
        bb_pedestrian_matrix = PedestrianBoundingBoxes.get_matrix(bb_transform)
        pedestrian_world_matrix = PedestrianBoundingBoxes.get_matrix(pedestrian.get_transform())
        bb_world_matrix = np.dot(pedestrian_world_matrix, bb_pedestrian_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = PedestrianBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix




# ==============================================================================
# -- VehicleBoundingBoxes ---------------------------------------------------
# ==============================================================================


class VehicleBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [VehicleBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, car_id):
        """
        Draws bounding boxes on pygame display.
        """
        ##bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        ##bb_surface.set_colorkey((0, 0, 0))

        if vehicle_bbox_record_list[car_id] == True:
            f = open(f'VehicleBBox/{car_id}/bbox{str(count_list[car_id])}', 'w')
            print(f'VehicleBBox/{car_id}/bbox{str(count_list[car_id])}')
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            if vehicle_bbox_record_list[car_id] == True:
                f.write(str(points)+"\n")
        
        if vehicle_bbox_record_list[car_id] == True:
            f.close()
            vehicle_bbox_record_list[car_id] = False
        
        
        
        ##display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = VehicleBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = VehicleBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = VehicleBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = VehicleBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = VehicleBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = VehicleBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = VehicleBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self, car_id):
        print (f'hey there! Im in BasicSynchronousClient {car_id}')
        self.client = None
        self.world = None
        self.camera = None
        self.camera_segmentation = None
        self.car = None

        self.display = None
        self.image = None
        self.segmentation_image = None

        self.capture = True
        self.capture_segmentation = True

        self.record = True
        self.seg_record = False
        self.rgb_record = False

        self.screen_capture = 0 
        self.loop_state = True 

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self, car_id):
        print(f'start of setup car of {car_id}')

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        while self.car == None:
            self.car = self.world.try_spawn_actor(car_bp, location)   
        self.car.set_autopilot(True)
        #tm = self.client.get_trafficmanager(8000)
        #tm.ignore_lights_percentage(self.car,100)

        print(f'end of setup car of {car_id}') 



    def setup_camera(self, car_id):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        x_value = 1.6
        seg_transform = carla.Transform(carla.Location(x=x_value, z=1.7), carla.Rotation(pitch=-15))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), seg_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image_seg: weak_self().set_segmentation(weak_self, image_seg, car_id))
        camera_transform = carla.Transform(carla.Location(x=x_value, z=1.7), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image, car_id))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration

    @staticmethod
    def set_image(weak_self, img, car_id):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

        if self.rgb_record:
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            ##lock.acquire()
            cv2.imwrite(f'custom_data/{car_id}/image' + str(self.image_count) + '.png', i3)
            ##lock.release()        
            print(f'custom_data/{car_id}/image' + str(self.image_count) + '.png')

    @staticmethod
    def set_segmentation(weak_self, img, car_id):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False


        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            ##lock.acquire()
            cv2.imwrite(f'SegmentationImage/{car_id}/seg' + str(self.image_count) +'.png', i3)
            ##lock.release()
            print(f'SegmentationImage/{car_id}/seg' + str(self.image_count) +'.png')

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            #surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            #display.blit(surface, (0, 0))

    def game_loop(self, car_id):
        """
        Main program loop.
        """
        try:
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.setup_car(car_id)
            self.setup_camera(car_id)
            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')
            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')


            self.image_count = 0
            self.time_interval = 0
            while True:
                self.time_interval += 1
                if ((self.time_interval % 10) == 0 and self.loop_state):
                    self.image_count = self.image_count + 1 
                    self.rgb_record = True
                    self.seg_record = True
                    vehicle_bbox_record_list[car_id] = True
                    pedestrian_bbox_record_list[car_id] = True
                    count_list[car_id] = self.image_count
                    print("-------------------------------------------------")
                    print("ImageCount - %d" %self.image_count)

                bounding_boxes = VehicleBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                pedestrian_bounding_boxes = PedestrianBoundingBoxes.get_bounding_boxes(pedestrians, self.camera)

                VehicleBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes, car_id)
                PedestrianBoundingBoxes.draw_bounding_boxes(self.display, pedestrian_bounding_boxes, car_id)
                
                self.rgb_record = False
                self.seg_record = False


        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.camera_segmentation.destroy()
            self.car.destroy()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def car_starter (car_id):
    client = BasicSynchronousClient(car_id)
    client.game_loop(car_id)

def main():
    """
    Initializes the client-side bounding box demo.
    """
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-l', '--CaptureLoop',
        metavar='N',
        default=10,
        type=int,
        help='set Capture Cycle settings, Recommand : above 100')
    argparser.add_argument(
        '-n', '--num-vehicles',
        metavar='NVP',
        default=12,
        type=int,
        help='Number of vehicles (default: 12)')

    arguments = argparser.parse_args()

    print(__doc__)

    try:
     
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
    
       
        for car_id in range (arguments.num_vehicles):
            print(f'car_id= {car_id}') 
            threading.Thread(target=car_starter, args=[car_id]).start()

        time.sleep(10)
        while True:
            world.tick()

    finally:
        print('EXIT')


if __name__ == '__main__':
    main()

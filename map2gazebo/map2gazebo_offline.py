import json
import cv2
import numpy as np
import trimesh
from matplotlib.tri import Triangulation
import yaml
import argparse
import os
import sys


class MapConverter():
    def __init__(self, map_dir, export_dir, threshold=105, height=2.0):

        self.threshold = threshold
        self.height = height
        self.export_dir = export_dir
        self.map_dir = map_dir

    def map_callback(self):
        map_array = cv2.imread(self.map_dir)
        height = map_array.shape[0]
        width = map_array.shape[1]
        map_array = cv2.flip(map_array, 0)
        print(f'loaded map file: {self.map_dir}, width: {width}, height: {height}')
        try:
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
        except cv2.error as err:
            print(err, "Conversion failed: Invalid image input, please check your file path")
            sys.exit()
        info_dir = self.map_dir.replace('png', 'yaml')
        with open(info_dir, 'r') as stream:
            map_info = yaml.load(stream, Loader=yaml.FullLoader)

        # set all -1 (unknown) values to 0 (unoccupied)
        map_array[map_array < 0] = 0
        obstacles = self.get_obstacles(map_array)
        print('Processing...')
        meshes = [self.contour_to_mesh(c, map_info) for c in obstacles]

        # corners = list(np.vstack(obstacles))
        # corners = [c[0] for c in corners]
        mesh = trimesh.util.concatenate(meshes)
        file_dir = os.path.join(self.export_dir, map_info['image'].replace('png', 'stl'))
        print(f'export file: {file_dir}')

        with open(file_dir, 'wb') as f:
            mesh.export(f, "stl")
        x_in_gazebo_world = map_info['origin'][0]
        y_in_gazebo_world = height * map_info['resolution'] + map_info['origin'][1]
        print(f'Gazebo model start position should be {x_in_gazebo_world} {y_in_gazebo_world}')

    def get_obstacles(self, map_array):
        """
        Get occupied regions of map
        """
        map_array = map_array.astype(np.uint8)
        # _, thresh_map = cv2.threshold(
        #     map_array, self.threshold, 100, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(
        #         thresh_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        obstacles = []
        height = map_array.shape[0]
        width = map_array.shape[1]
        print(f'height -> {height}')
        print(f'width -> {width}')
        ssss = []
        for row in range(height):
            for col in range(width):
                if map_array[row][col] not in ssss:
                    ssss.append(map_array[row][col])
                if map_array[row][col] <= 100:
                    obstacles.append((col, row))
        print(ssss)
        # Using cv2.RETR_CCOMP classifies external contours at top level of
        # hierarchy and interior contours at second level.
        # If the whole space is enclosed by walls RETR_EXTERNAL will exclude
        # all interior obstacles e.g. furniture.
        # https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        # hierarchy = hierarchy[0]
        # output_contours = []
        # for idx, contour in enumerate(obstacles):
        #     output_contours.append(contour) if 0 not in contour else print('Remove image boundary')

        return obstacles

    def contour_to_mesh(self, coordinate, metadata):
        height = np.array([0, 0, self.height])
        meshes = []
        x, y = coordinate
        vertices = []
        new_vertices = [
            self.coords_to_loc((x, y), metadata),
            self.coords_to_loc((x, y + 1), metadata),
            self.coords_to_loc((x + 1, y), metadata),
            self.coords_to_loc((x + 1, y + 1), metadata)]
        vertices.extend(new_vertices)
        vertices.extend([v + height for v in new_vertices])
        faces = [[0, 2, 4],
                 [4, 2, 6],
                 [1, 2, 0],
                 [3, 2, 1],
                 [5, 0, 4],
                 [1, 0, 5],
                 [3, 7, 2],
                 [7, 6, 2],
                 [7, 4, 6],
                 [5, 4, 7],
                 [1, 5, 3],
                 [7, 3, 5]]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if not mesh.is_volume:
            mesh.fix_normals()
        meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        mesh.update_faces(mesh.unique_faces())
        # mesh will still have internal faces.  Would be better to get
        # all duplicate faces and remove both of them, since duplicate faces
        # are guaranteed to be internal faces
        return mesh

    def coords_to_loc(self, coords, metadata):
        x, y = coords
        loc_x = x * metadata['resolution'] + metadata['origin'][0]
        loc_y = y * metadata['resolution'] + metadata['origin'][1]
        # TODO: transform (x*res, y*res, 0.0) by Pose map_metadata.origin
        # instead of assuming origin is at z=0 with no rotation wrt map frame
        return np.array([loc_x, loc_y, 0.0])


if __name__ == "__main__":
    # --map_dir  --export_dir /opt/ros/humble/src/cloudminds/ginger_robot/multi_robot_simulation/models/SOHO36F/meshes
    ginger_maps = '/home/ginger/ginger_maps/release'
    with open(os.path.join(ginger_maps, 'maplist.json'), "rt") as f:
        map_lists_str = f.read()
    map_lists = json.loads(map_lists_str)
    map_png_path = os.path.join(ginger_maps, map_lists['currmap']['uuid'], map_lists['currmap']['name'], 'grid_map_layer', 'map.png')
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--map_dir', type=str,
        default=map_png_path,
        help='File name of the map to convert'
    )

    parser.add_argument(
        '--export_dir', type=str,
        default='/opt/ros/humble/src/cloudminds/ginger_robot/multi_robot_simulation/models/SOHO36F/meshes',
        help='Mesh output directory'
    )

    option = parser.parse_args()

    Converter = MapConverter(option.map_dir, option.export_dir)
    Converter.map_callback()
    print('Conversion Done')

from abc import ABC, abstractmethod

import taichi as ti
import taichi.math as tm

from .geometry import Geometry
from .materials import Material
from .ray import Ray, HitData


@ti.data_oriented
class RayIntersector(ABC):

    def __init__(self, geometry: Geometry):
        self.EPSILON = 1e-7
        self.geometry = geometry

    @abstractmethod
    @ti.func
    def query_ray(ray: Ray) -> HitData:
        pass

    @ti.func
    def intersect_triangle(self, ray: Ray, triangle_id: int) -> HitData:

        hit_data = HitData()

        # Grab Vertices
        vert_ids = (
            self.geometry.triangle_vertex_ids[triangle_id - 1] - 1
        )  # Vertices are indexed from 1
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # Normals at each vertex
        normal_indices = self.geometry.triangle_normal_ids[triangle_id - 1] - 1

        normal_0 = self.geometry.normals[normal_indices[0]]
        normal_1 = self.geometry.normals[normal_indices[1]]
        normal_2 = self.geometry.normals[normal_indices[2]]

        # Material of the triangle
        material_id = self.geometry.triangle_material_ids[triangle_id - 1]

        """
        Ray-Triangle intersection
        
        - v0, v1, v2 are the vertices of a given triangle
        - normal_0, normal_1, normal2 are the normals at each vertex

        # check if the ray intersects inside the given triangle
        # if it is, fill the hit_data with the following information
        """

        # initialize hit data member variables
        is_hit = False
        is_backfacing = False
        distance = 0.0
        barycentric_coords = tm.vec2([0.0, 0.0])
        normal = tm.vec3([0.0, 0.0, 0.0])

        # compute edge vectors e1, e2
        e1 = v1 - v0
        e2 = v2 - v0

        # compute the determinant
        det = tm.dot(e1, tm.cross(ray.direction, e2))

        # check if ray is parallel to triangle (det = 0)
        # if ray is not parallel, proceed:
        if abs(det) > self.EPSILON:

            # compute barycentric coordinates
            u = (1 / det) * tm.dot(ray.origin - v0, tm.cross(ray.direction, e2))
            v = (1 / det) * tm.dot(ray.direction, tm.cross(ray.origin - v0, e1))
            w = 1 - u - v

            # verify barycentric coordinates within sheared triangle bounds
            if (0 <= u <= 1) and (0 <= v <= 1) and (u + v <= 1):

                # compute distance t where ray intersects triangle
                t = (1 / det) * tm.dot(e2, tm.cross(ray.origin - v0, e1))

                # only consider intersections for t > 0
                if t > self.EPSILON:
                    is_hit = True
                    if det < 0:
                        is_backfacing = True
                    distance = t
                    barycentric_coords = tm.vec2([u, v])

                    if is_backfacing:
                        normal_0 = -1 * normal_0
                        normal_1 = -1 * normal_1
                        normal_2 = -1 * normal_2
                    normal = tm.normalize(
                        (u * normal_0) + (v * normal_1) + (w * normal_2)
                    )

        # populate and return HitData
        hit_data.is_hit = is_hit
        hit_data.is_backfacing = is_backfacing
        hit_data.triangle_id = triangle_id
        hit_data.distance = distance
        hit_data.barycentric_coords = barycentric_coords
        hit_data.normal = normal
        hit_data.material_id = material_id

        return hit_data


@ti.data_oriented
class BruteForceRayIntersector(RayIntersector):

    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    @ti.func
    def query_ray(self, ray: Ray) -> HitData:

        closest_hit = HitData()
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            hit_data = self.intersect_triangle(ray, triangle_id)

            if hit_data.is_hit:
                if (hit_data.distance < closest_hit.distance) or (
                    not closest_hit.is_hit
                ):
                    closest_hit = hit_data

        return closest_hit

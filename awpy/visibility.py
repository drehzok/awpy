"""Module for calculating visibility.

Reference: https://github.com/AtomicBool/cs2-map-parser
"""

from __future__ import annotations

import pathlib
import struct
import numba
import numpy as np
from dataclasses import dataclass
from typing import Literal, overload

from loguru import logger

import awpy.vector

LEAF = 4

@dataclass
class Triangle:
    """A triangle in 3D space defined by three vertices.

    Attributes:
        p1: First vertex of the triangle.
        p2: Second vertex of the triangle.
        p3: Third vertex of the triangle.
    """

    p1: awpy.vector.Vector3
    p2: awpy.vector.Vector3
    p3: awpy.vector.Vector3

    def get_centroid(self) -> awpy.vector.Vector3:
        """Calculate the centroid of the triangle.

        Returns:
            awpy.vector.Vector3: Centroid of the triangle.
        """
        return awpy.vector.Vector3(
            (self.p1.x + self.p2.x + self.p3.x) / 3,
            (self.p1.y + self.p2.y + self.p3.y) / 3,
            (self.p1.z + self.p2.z + self.p3.z) / 3,
        )


@dataclass
class Edge:
    """An edge in a triangulated mesh.

    Attributes:
        next: Index of the next edge in the face.
        twin: Index of the twin edge in the adjacent face.
        origin: Index of the vertex where this edge starts.
        face: Index of the face this edge belongs to.
    """

    next: int
    twin: int
    origin: int
    face: int


class KV3Parser:
    """Parser for KV3 format files used in Source 2 engine.

    This class provides functionality to parse KV3 files, which are used to store
    various game data including physics collision meshes.

    Attributes:
        content: Raw content of the KV3 file.
        index: Current parsing position in the content.
        parsed_data: Resulting parsed data structure.
    """

    def __init__(self) -> None:
        """Initialize a new KV3Parser instance."""
        self.content = ""
        self.index = 0
        self.parsed_data = None

    def parse(self, content: str) -> None:
        """Parse the given KV3 content string.

        Args:
            content: String containing KV3 formatted data.
        """
        self.content = content
        self.index = 0
        self._skip_until_first_bracket()
        self.parsed_data = self._parse_value()

    def get_value(self, path: str) -> str:
        """Get a value from the parsed data using a dot-separated path.

        Args:
            path: Dot-separated path to the desired value, e.g.,
                "section.subsection[0].value"

        Returns:
            String value at the specified path, or empty string
                if not found.
        """
        if not self.parsed_data:
            return ""

        current = self.parsed_data
        for segment in path.split("."):
            key = segment
            array_index = None

            if "[" in segment:
                key = segment[: segment.find("[")]
                array_index = int(segment[segment.find("[") + 1 : segment.find("]")])

            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return ""

            if array_index is not None:
                if isinstance(current, list) and array_index < len(current):
                    current = current[array_index]
                else:
                    return ""

        return current if isinstance(current, str) else ""

    def _skip_until_first_bracket(self) -> None:
        """Skip content until the first opening bracket is found."""
        while self.index < len(self.content) and self.content[self.index] != "{":
            self.index = self.content.find("\n", self.index) + 1

    def _skip_whitespace(self) -> None:
        """Skip all whitespace characters at the current position."""
        while self.index < len(self.content) and self.content[self.index].isspace():
            self.index += 1

    def _parse_value(self) -> dict | list | str | None:
        """Parse a value from the current position.

        Returns:
            Parsed value which can be a dictionary, list, or string,
                or None if parsing fails.
        """
        self._skip_whitespace()
        if self.index >= len(self.content):
            return None

        char = self.content[self.index]
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char == "#" and self.index + 1 < len(self.content) and self.content[self.index + 1] == "[":
            self.index += 1
            return self._parse_byte_array()
        return self._parse_string()

    def _parse_object(self) -> dict:
        """Parse a KV3 object starting at the current position.

        Returns:
            Dictionary containing the parsed key-value pairs.
        """
        self.index += 1  # Skip {
        obj = {}
        while self.index < len(self.content):
            self._skip_whitespace()
            if self.content[self.index] == "}":
                self.index += 1
                return obj

            key = self._parse_string()
            self._skip_whitespace()
            if self.content[self.index] == "=":
                self.index += 1

            value = self._parse_value()
            if key and value is not None:
                obj[key] = value

            self._skip_whitespace()
            if self.content[self.index] == ",":
                self.index += 1

        return obj

    def _parse_array(self) -> list:
        """Parse a KV3 array starting at the current position.

        Returns:
            List containing the parsed values.
        """
        self.index += 1  # Skip [
        arr = []
        while self.index < len(self.content):
            self._skip_whitespace()
            if self.content[self.index] == "]":
                self.index += 1
                return arr

            value = self._parse_value()
            if value is not None:
                arr.append(value)

            self._skip_whitespace()
            if self.content[self.index] == ",":
                self.index += 1
        return arr

    def _parse_byte_array(self) -> str:
        """Parse a KV3 byte array starting at the current position.

        Returns:
            Space-separated string of byte values.
        """
        self.index += 1  # Skip [
        start = self.index
        while self.index < len(self.content) and self.content[self.index] != "]":
            self.index += 1
        byte_str = self.content[start : self.index].strip()
        self.index += 1  # Skip ]
        return " ".join(byte_str.split())

    def _parse_string(self) -> str:
        """Parse a string value at the current position.

        Returns:
            Parsed string value.
        """
        start = self.index
        while self.index < len(self.content):
            char = self.content[self.index]
            if char in "={}[], \n":
                break
            self.index += 1
        return self.content[start : self.index].strip()


class VphysParser:
    """Parser for VPhys collision files.

    This class extracts and processes collision geometry data
        from VPhys files, converting it into a set of triangles.

    Attributes:
        vphys_file (Path): Path to the VPhys file.
        triangles (list[Triangle]): List of parsed triangles from the VPhys file.
        kv3_parser (KV3Parser): Helper parser for extracting key-value data from
            the .vphys file.
    """

    def __init__(self, vphys_file: str | pathlib.Path) -> None:
        """Initializes the parser with the path to a VPhys file.

        Args:
            vphys_file (str | pathlib.Path): Path to the VPhys file
                to parse.
        """
        self.vphys_file = pathlib.Path(vphys_file)
        self.triangles: list[Triangle] = []
        self.kv3_parser = KV3Parser()
        self.parse()

    @overload
    @staticmethod
    def bytes_to_vec(byte_str: str, element_type: Literal["uint8", "int32"]) -> list[int]: ...

    @overload
    @staticmethod
    def bytes_to_vec(byte_str: str, element_type: Literal["float"]) -> list[float]: ...

    @staticmethod
    def bytes_to_vec(byte_str: str, element_type: Literal["uint8", "int32", "float"]) -> list[int] | list[float]:
        """Converts a space-separated string of byte values into a list of numbers.

        Args:
            byte_str (str): Space-separated string of hexadecimal byte values.
            element_type (int): Types represented by the bytes (uint8, int32, float).

        Returns:
            list[int | float]: List of converted values (integers for
                uint8, floats for size 4).
        """
        bytes_list = [int(b, 16) for b in byte_str.split()]
        result = []

        if element_type == "uint8":
            return bytes_list

        element_size = 4  # For int and float

        # Convert bytes to appropriate type based on size
        for i in range(0, len(bytes_list), element_size):
            chunk = bytes(bytes_list[i : i + element_size])
            if element_type == "float":  # float
                val = struct.unpack("f", chunk)[0]
                result.append(val)
            else:  # int32
                val = struct.unpack("i", chunk)[0]
                result.append(val)
        return result

    def get_collision_attribute_indices_for_default_group(self) -> list[str]:
        """Get collision attribute indices for the default group.

        Returns:
            list[int]: List of collision attribute indices for the default group.
        """
        collision_attribute_indices = []
        idx = 0
        while True:
            collision_group_string = self.kv3_parser.get_value(f"m_collisionAttributes[{idx}].m_CollisionGroupString")
            if not collision_group_string:
                break
            if collision_group_string.lower() == '"default"':
                collision_attribute_indices.append(str(idx))
            idx += 1
        return collision_attribute_indices

    def parse(self) -> None:
        """Parses the VPhys file and extracts collision geometry.

        Processes hulls and meshes in the VPhys file to generate a list of triangles.
        """
        if len(self.triangles) > 0:
            logger.debug(f"VPhys data already parsed, got {len(self.triangles)} triangles.")
            return

        logger.debug(f"Parsing vphys file: {self.vphys_file}")

        # Read file
        with open(self.vphys_file) as f:
            data = f.read()

        # Parse VPhys data
        self.kv3_parser.parse(data)

        collision_attribute_indices = self.get_collision_attribute_indices_for_default_group()

        logger.debug(f"Extracted collision attribute indices: {collision_attribute_indices}")

        # Process hulls
        hull_idx = 0
        hull_count = 0
        while True:
            if hull_idx % 1000 == 0:
                logger.debug(f"Processing hull {hull_idx}...")

            collision_idx = self.kv3_parser.get_value(
                f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_nCollisionAttributeIndex"
            )
            if not collision_idx:
                break

            if collision_idx in collision_attribute_indices:
                # Get vertices
                vertex_str = self.kv3_parser.get_value(
                    f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_VertexPositions"
                )
                if not vertex_str:
                    vertex_str = self.kv3_parser.get_value(
                        f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_Vertices"
                    )

                vertex_data = self.bytes_to_vec(vertex_str, "float")
                vertices = [
                    awpy.vector.Vector3(vertex_data[i], vertex_data[i + 1], vertex_data[i + 2])
                    for i in range(0, len(vertex_data), 3)
                ]

                # Get faces and edges
                faces = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_Faces"),
                    "uint8",
                )
                edge_data = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_Edges"),
                    "uint8",
                )

                edges = [
                    Edge(
                        edge_data[i],
                        edge_data[i + 1],
                        edge_data[i + 2],
                        edge_data[i + 3],
                    )
                    for i in range(0, len(edge_data), 4)
                ]

                # Process triangles
                for start_edge in faces:
                    edge = edges[start_edge].next
                    while edge != start_edge:
                        next_edge = edges[edge].next
                        self.triangles.append(
                            Triangle(
                                vertices[edges[start_edge].origin],
                                vertices[edges[edge].origin],
                                vertices[edges[next_edge].origin],
                            )
                        )
                        edge = next_edge

                hull_count += 1
            hull_idx += 1

        # Process meshes
        mesh_idx = 0
        mesh_count = 0
        while True:
            logger.debug(f"Processing mesh {mesh_idx}...")
            collision_idx = self.kv3_parser.get_value(
                f"m_parts[0].m_rnShape.m_meshes[{mesh_idx}].m_nCollisionAttributeIndex"
            )
            if not collision_idx:
                break

            if collision_idx in collision_attribute_indices:
                # Get triangles and vertices
                tri_data = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_meshes[{mesh_idx}].m_Mesh.m_Triangles"),
                    "int32",
                )
                vertex_data = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_meshes[{mesh_idx}].m_Mesh.m_Vertices"),
                    "float",
                )

                vertices = [
                    awpy.vector.Vector3(vertex_data[i], vertex_data[i + 1], vertex_data[i + 2])
                    for i in range(0, len(vertex_data), 3)
                ]

                for i in range(0, len(tri_data), 3):
                    self.triangles.append(
                        Triangle(
                            vertices[int(tri_data[i])],
                            vertices[int(tri_data[i + 1])],
                            vertices[int(tri_data[i + 2])],
                        )
                    )

                mesh_count += 1
            mesh_idx += 1

    def to_tri(self, path: str | pathlib.Path | None) -> None:
        """Export parsed triangles to a .tri file.

        Args:
            path: Path to the output .tri file.
        """
        if not path:
            path = self.vphys_file.with_suffix(".tri")
        outpath = pathlib.Path(path)

        logger.debug(f"Exporting {len(self.triangles)} triangles to {outpath}")
        with open(outpath, "wb") as f:
            for triangle in self.triangles:
                # Write all awpy.vector.Vector3 components as float32
                f.write(struct.pack("f", triangle.p1.x))
                f.write(struct.pack("f", triangle.p1.y))
                f.write(struct.pack("f", triangle.p1.z))
                f.write(struct.pack("f", triangle.p2.x))
                f.write(struct.pack("f", triangle.p2.y))
                f.write(struct.pack("f", triangle.p2.z))
                f.write(struct.pack("f", triangle.p3.x))
                f.write(struct.pack("f", triangle.p3.y))
                f.write(struct.pack("f", triangle.p3.z))

        logger.success(f"Processed {len(self.triangles)} triangles from {self.vphys_file} -> {outpath}")





@numba.njit(inline="always")
def aabb_hit(o0, o1, o2, d0, d1, d2, inv0, inv1, inv2,
             bmin0, bmin1, bmin2, bmax0, bmax1, bmax2, seg_tmax):
    # Handle near-zero direction by explicit slab containment checks
    eps = 1e-9

    if abs(d0) < eps:
        if o0 < bmin0 or o0 > bmax0:
            return False
        tmin0 = -1e30
        tmax0 =  1e30
    else:
        t1 = (bmin0 - o0) * inv0
        t2 = (bmax0 - o0) * inv0
        tmin0 = t1 if t1 < t2 else t2
        tmax0 = t2 if t1 < t2 else t1

    if abs(d1) < eps:
        if o1 < bmin1 or o1 > bmax1:
            return False
        tmin1 = -1e30
        tmax1 =  1e30
    else:
        t1 = (bmin1 - o1) * inv1
        t2 = (bmax1 - o1) * inv1
        tmin1 = t1 if t1 < t2 else t2
        tmax1 = t2 if t1 < t2 else t1

    if abs(d2) < eps:
        if o2 < bmin2 or o2 > bmax2:
            return False
        tmin2 = -1e30
        tmax2 =  1e30
    else:
        t1 = (bmin2 - o2) * inv2
        t2 = (bmax2 - o2) * inv2
        tmin2 = t1 if t1 < t2 else t2
        tmax2 = t2 if t1 < t2 else t1

    t_enter = tmin0
    if tmin1 > t_enter: t_enter = tmin1
    if tmin2 > t_enter: t_enter = tmin2

    t_exit = tmax0
    if tmax1 < t_exit: t_exit = tmax1
    if tmax2 < t_exit: t_exit = tmax2

    if t_enter > t_exit:
        return False
    if t_exit < 0.0:
        return False
    if t_enter > seg_tmax:
        return False
    return True


@numba.njit(inline="always")
def ray_tri_hit(o0, o1, o2, d0, d1, d2,
                p10, p11, p12,
                p20, p21, p22,
                p30, p31, p32,
                seg_tmax):
    eps = 1e-6

    # edge1 = p2 - p1
    e10 = p20 - p10
    e11 = p21 - p11
    e12 = p22 - p12

    # edge2 = p3 - p1
    e20 = p30 - p10
    e21 = p31 - p11
    e22 = p32 - p12

    # h = d x edge2
    h0 = d1 * e22 - d2 * e21
    h1 = d2 * e20 - d0 * e22
    h2 = d0 * e21 - d1 * e20

    a = e10 * h0 + e11 * h1 + e12 * h2
    if -eps < a < eps:
        return False

    f = 1.0 / a

    # s = o - p1
    s0 = o0 - p10
    s1 = o1 - p11
    s2 = o2 - p12

    u = f * (s0 * h0 + s1 * h1 + s2 * h2)
    if u < 0.0 or u > 1.0:
        return False

    # q = s x edge1
    q0 = s1 * e12 - s2 * e11
    q1 = s2 * e10 - s0 * e12
    q2 = s0 * e11 - s1 * e10

    v = f * (d0 * q0 + d1 * q1 + d2 * q2)
    if v < 0.0 or (u + v) > 1.0:
        return False

    t = f * (e20 * q0 + e21 * q1 + e22 * q2)
    return (t > eps) and (t <= seg_tmax)

@numba.njit(parallel=True, fastmath=False, cache=True)
def is_visible_batch_numba(
    starts, ends,
    aabb_min, aabb_max,
    left, right,
    tri_offset, tri_count, tri_indices,
    p1, p2, p3,
    root,
    stack_buf,
):
    R = starts.shape[0]
    out = np.empty(R, dtype=np.uint8)

    # Conservative stack depth: for leaf=1, depth <= 2N, but practical depth << N.
    # To be safe, allocate a stack sized to a fixed upper bound.
    # A good compromise: 128 or 256 often works for balanced BVHs; if you overflow, increase.
    STACK_MAX = stack_buf.shape[1]

    for i in numba.prange(R):
        o0, o1, o2 = starts[i, 0], starts[i, 1], starts[i, 2]
        e0, e1, e2 = ends[i, 0], ends[i, 1], ends[i, 2]

        dir0 = e0 - o0
        dir1 = e1 - o1
        dir2 = e2 - o2

        dist = (dir0*dir0 + dir1*dir1 + dir2*dir2) ** 0.5
        if dist < 1e-6:
            out[i] = 1
            continue

        inv_dist = 1.0 / dist
        d0 = dir0 * inv_dist
        d1 = dir1 * inv_dist
        d2 = dir2 * inv_dist

        inv0 = 1.0 / d0 if abs(d0) >= 1e-9 else 1e30
        inv1 = 1.0 / d1 if abs(d1) >= 1e-9 else 1e30
        inv2 = 1.0 / d2 if abs(d2) >= 1e-9 else 1e30


        stack = stack_buf[i]
        sp = 0
        stack[sp] = root
        sp += 1

        occluded = False

        while sp > 0 and not occluded:
            sp -= 1
            node = stack[sp]

            bmin0 = aabb_min[node, 0]; bmin1 = aabb_min[node, 1]; bmin2 = aabb_min[node, 2]
            bmax0 = aabb_max[node, 0]; bmax1 = aabb_max[node, 1]; bmax2 = aabb_max[node, 2]

            if not aabb_hit(o0,o1,o2, d0,d1,d2, inv0,inv1,inv2, bmin0,bmin1,bmin2, bmax0,bmax1,bmax2, dist):
                continue

            l = left[node]
            r = right[node]

            if l == -1 and r == -1:
                off = tri_offset[node]
                cnt = tri_count[node]
                # LEAF=1 -> cnt should be 1
                for k in range(cnt):
                    tid = tri_indices[off + k]

                    if ray_tri_hit(
                        o0,o1,o2, d0,d1,d2,
                        p1[tid,0], p1[tid,1], p1[tid,2],
                        p2[tid,0], p2[tid,1], p2[tid,2],
                        p3[tid,0], p3[tid,1], p3[tid,2],
                        dist
                    ):
                        occluded = True
                        break
            else:
                # Push children; order can matter for early-out but keep simple
                # Guard against stack overflow
                if sp + 2 <= STACK_MAX:
                    stack[sp] = l; sp += 1
                    stack[sp] = r; sp += 1
                else:
                    # Fallback: if overflow, mark not visible conservatively or increase STACK_MAX
                    # Better: increase STACK_MAX; for now we bail out as visible=False to avoid false positives.
                    occluded = True

        out[i] = 0 if occluded else 1

    return out



class BVH:
    def __init__(self, data):
        self.data = data
        N = int(data["centroid"].shape[0])
        self.N = N
        self.tri_indices = np.arange(N, dtype=np.int32)

        max_nodes = 2 * N  

        self.aabb_min = np.empty((max_nodes, 3), dtype=np.float32)
        self.aabb_max = np.empty((max_nodes, 3), dtype=np.float32)
        self.left = np.full((max_nodes,), -1, dtype=np.int32)
        self.right = np.full((max_nodes,), -1, dtype=np.int32)
        self.tri_offset = np.full((max_nodes,), -1, dtype=np.int32)
        self.tri_count = np.zeros((max_nodes,), dtype=np.int32)

        self.node_count = 0
        self.root = self._build(0, N)

        M = self.node_count
        self.aabb_min = self.aabb_min[:M]
        self.aabb_max = self.aabb_max[:M]
        self.left = self.left[:M]
        self.right = self.right[:M]
        self.tri_offset = self.tri_offset[:M]
        self.tri_count = self.tri_count[:M]

    def _alloc_node(self):
        idx = self.node_count
        # Safety check (gives a clearer error than NumPy IndexError)
        if idx >= self.aabb_min.shape[0]:
            raise RuntimeError(
                f"BVH node buffer overflow: idx={idx}, cap={self.aabb_min.shape[0]}. "
                f"This implies an empty-range split or non-binary leaf creation."
            )
        self.node_count += 1
        return idx

    def _build(self, lo, hi):
        # Hard guard: never build empty ranges
        if hi <= lo:
            raise RuntimeError(f"Empty range in BVH build: lo={lo}, hi={hi}")

        node = self._alloc_node()
        sl = self.tri_indices[lo:hi]
        n = hi - lo

        tri_min = self.data["tri_aabb_min"][sl]
        tri_max = self.data["tri_aabb_max"][sl]
        self.aabb_min[node] = tri_min.min(axis=0)
        self.aabb_max[node] = tri_max.max(axis=0)

        if n <= LEAF:
            self.tri_offset[node] = lo
            self.tri_count[node] = n
            return node

        # Choose split axis by centroid spread
        cent = self.data["centroid"][sl]
        spread = cent.max(axis=0) - cent.min(axis=0)
        axis = int(np.argmax(spread))

        mid = lo + (n // 2)

        # Guard bad splits (shouldn't happen for n>=2, but protects you from edge cases)
        if mid <= lo or mid >= hi:
            self.tri_offset[node] = lo
            self.tri_count[node] = n
            return node

        # Partition indices by centroid coordinate along axis
        # (Uses cent computed above to avoid re-indexing)
        keys = cent[:, axis]
        part = np.argpartition(keys, n // 2)
        sl[:] = sl[part]

        left_node = self._build(lo, mid)
        right_node = self._build(mid, hi)
        self.left[node] = left_node
        self.right[node] = right_node
        return node


class VisibilityChecker:
    """Class for visibility checking in 3D space using a BVH structure."""

    def __init__(self, path: pathlib.Path | None = None) -> None:
        """Initialize the visibility checker with a list of triangles.

        Args:
            path (pathlib.Path | None, optional): Path to a .tri file to read
                triangles from.
            triangles (list[Triangle] | None, optional): List of triangles to
                build the BVH from.
        """
        if path is not None:
            self.data = self.read_tri_file(path)

        # maybe do something like if triangles - list of Triangle then covert to numpy array + compute centroids?
        self.n_triangles = len(self.data['p1'])
        self.bvh = BVH(self.data)

    def __repr__(self) -> str:
        """Return a string representation of the VisibilityChecker."""
        return f"VisibilityChecker(n_triangles={self.n_triangles})"

    def is_visible_batch_fast(self, starts, ends):
        bvh = self.bvh
        starts = np.ascontiguousarray(starts, dtype=np.float32)
        ends   = np.ascontiguousarray(ends,   dtype=np.float32)

        R = starts.shape[0]
        STACK_MAX = 256  # choose
        stack_buf = np.empty((R, STACK_MAX), dtype=np.int32)
        # Ensure triangle arrays are float32
        p1 = np.asarray(bvh.data["p1"], dtype=np.float32)
        p2 = np.asarray(bvh.data["p2"], dtype=np.float32)
        p3 = np.asarray(bvh.data["p3"], dtype=np.float32)

        vis_u8 = is_visible_batch_numba(
            starts, ends,
            bvh.aabb_min, bvh.aabb_max,
            bvh.left, bvh.right,
            bvh.tri_offset, bvh.tri_count, bvh.tri_indices,
            p1, p2, p3,
            np.int32(bvh.root),
            stack_buf,
        )
        return vis_u8.astype(bool)

    @staticmethod
    def read_tri_file(tri_file: str | pathlib.Path):
        tri_file = pathlib.Path(tri_file)

        data = np.memmap(tri_file, dtype=np.float32, mode="r")
        data = np.asarray(data).reshape(-1, 9)

        p1 = data[:, 0:3]
        p2 = data[:, 3:6]
        p3 = data[:, 6:9]

        centroid = (p1 + p2 + p3) * (1.0 / 3.0)
        tri_aabb_min = np.minimum(p1, np.minimum(p2, p3))
        tri_aabb_max = np.maximum(p1, np.maximum(p2, p3))

        return {
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "centroid": centroid,
            "tri_aabb_min": tri_aabb_min,
            "tri_aabb_max": tri_aabb_max,
        }


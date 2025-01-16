import random
import math
import pygame
import numpy as np
from scipy.ndimage import binary_dilation
import heapq

class KDTree(object):
    def __init__(self, points, dim, dist_sq_func=None):
        if dist_sq_func is None:
            dist_sq_func = lambda a, b: (a[0] - b.x) ** 2 + (a[1] - b.y) ** 2

        def getXorY(node, i):
            return node.x * (not i) + node.y * i 
                
        def make(points, i=0):
            if len(points) > 1:
                points = sorted(points, key=lambda p: getXorY(p[1], i))
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), 
                    points[m]]
            if len(points) == 1:
                return [None, None, list(points)[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = getXorY(node[2][1], i) - getXorY(point[1], i)
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2][1]) if not (node[2][1].x == point[0] and node[2][1].y == point[1]) else float('inf')
                dx = getXorY(node[2][1], i) - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2][1], node[2][0]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2][1], node[2][0]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq, 
                        heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2], h[3]) if return_dist_sq else (h[2], h[3]) 
                    for h in sorted(heap)][::-1]

        def walk(node):
            if node is not None:
                for j in 0, 1:
                    for x in walk(node[j]):
                        yield x
                yield node[2]

        self._add_point = add_point
        self._get_knn = get_knn 
        self._root = make(points)
        self._walk = walk

    def __iter__(self):
        return self._walk(self._root)
        
    def add_point(self, point):
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None

class RRTMap:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.maph, self.mapw = self.MapDimensions

        self.MapWindowName = "RRT* - Optimized Path Planning"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.fill((255, 255, 255))
        self.nodeRad = 0
        self.nodeThickness = 0
        self.edgeThickness = 1

        self.obstacles = []
        self.obsDim = obsdim
        self.obsNum = obsnum

        self.black = (0, 0, 0)
        self.grey = (75, 75, 75)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)

    def drawMap(self, obstacles):
        pygame.draw.circle(self.map, self.Red, self.start, self.nodeRad + 5, 0)
        pygame.draw.circle(self.map, self.Green, self.goal, self.nodeRad + 5, 0)
        self.drawObs(obstacles)

    def drawPath(self, path, from_goal = False):
        color = self.Red if not from_goal else (255, 165, 0)
        for i in range(len(path)):
            # pygame.draw.circle(self.map, color, path[i], self.nodeRad + 4, 0)
            if i < len(path) - 1:
                pygame.draw.line(self.map, color, path[i], path[i + 1], self.nodeRad +  4)

    def resetPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.Blue, node, self.nodeRad+4, 0)

    
    def makeRandomShape(self):
        shape_type = random.choice(["rect", "circle", "triangle"])

        if shape_type == "rect":
            x = random.randint(0, self.mapw - 30)
            y = random.randint(0, self.maph - 30)
            return "rect", pygame.Rect((x, y), (random.randint(30, self.obsDim), random.randint(30, self.obsDim)))

        elif shape_type == "circle":
            x = random.randint(self.obsDim, self.mapw - 30)
            y = random.randint(self.obsDim, self.maph - 30)
            radius = random.randint(30, self.obsDim) // 2
            return "circle", ((x, y), radius)

        elif shape_type == "triangle":
            x1 = random.randint(0, self.mapw - 30)
            y1 = random.randint(0, self.maph - 30)
            x2 = x1 + random.randint(30, self.obsDim)
            y2 = y1
            x3 = x1 + random.randint(30, self.obsDim) // 2
            y3 = y1 - random.randint(30, self.obsDim)
            return "triangle", [(x1, y1), (x2, y2), (x3, y3)]

    def makeobs(self):
        obs = []
        
        # Create rectangles representing the goal and start as circles with radius 5
        goal_circle = pygame.Rect(self.goal[0] - 5, self.goal[1] - 5, 10, 10)  # Circle with radius 5
        start_circle = pygame.Rect(self.start[0] - 5, self.start[1] - 5, 10, 10)  # Circle with radius 5

        for _ in range(self.obsNum):
            shape = None
            startGoalCol = True
            while startGoalCol:
                shape = self.makeRandomShape()
                shape_type, properties = shape
                if shape_type == "rect":
                    # Rectangle properties (rectangular shape)
                    rect = properties
                    startGoalCol = rect.colliderect(goal_circle) or rect.colliderect(start_circle)
                elif shape_type == "circle":
                    # Circle properties (circle shape)
                    circle_center, radius = properties
                    circle_rect = pygame.Rect(circle_center[0] - radius, circle_center[1] - radius, 2 * radius, 2 * radius)
                    startGoalCol = circle_rect.colliderect(goal_circle) or circle_rect.colliderect(start_circle)
                elif shape_type == "triangle":
                    # Triangle properties (triangle shape)
                    triangle_rect = pygame.Rect(min(p[0] for p in properties), min(p[1] for p in properties),
                                                self.obsDim, self.obsDim)
                    startGoalCol = triangle_rect.colliderect(goal_circle) or triangle_rect.colliderect(start_circle)

            obs.append(shape)

        self.obstacles = obs.copy()
        return obs
    

    def drawObs(self, obstacles):
        for obs in obstacles:
            shape, properties = obs
            if shape == "rect":
                pygame.draw.rect(self.map, self.black, properties)
            elif shape == "circle":
                pygame.draw.circle(self.map, self.black, properties[0], properties[1])
            elif shape == "triangle":
                pygame.draw.polygon(self.map, self.black, properties)          

class Ellipse:
    def __init__(self, focus1, focus2, major_axis_length, maxX, maxY, obstacle_grid):
        self.maxX = maxX
        self.maxY = maxY
        self.obstacle_grid = obstacle_grid

        self.focus1 = focus1
        self.focus2 = focus2
        
        self.center_x = (focus1[0] + focus2[0]) / 2
        self.center_y = (focus1[1] + focus2[1]) / 2
        
        self.a = major_axis_length / 2
        
        self.c = math.sqrt(
            (focus2[0] - focus1[0])**2 + 
            (focus2[1] - focus1[1])**2
        ) / 2
        
        self.b = math.sqrt(self.a**2 - self.c**2)

        self.rotation = math.atan2(
            focus2[1] - focus1[1], 
            focus2[0] - focus1[0]
        )

    def get_area(self):
        return math.pi * self.a * self.b

    def generate_random_point(self):
        while True:
            r = math.sqrt(random.uniform(0, 1))
            theta = random.uniform(0, 2 * math.pi)
            
            x = r * self.a * math.cos(theta)
            y = r * self.b * math.sin(theta)
            
            x_rotated = (
                x * math.cos(self.rotation) - 
                y * math.sin(self.rotation)
            )
            y_rotated = (
                x * math.sin(self.rotation) + 
                y * math.cos(self.rotation)
            )

            final_x = round(x_rotated + self.center_x)
            final_y = round(y_rotated + self.center_y)

            if 0 < final_x < self.maxX and 0 < final_y < self.maxY and not self.obstacle_grid[final_x, final_y]:
                return round(final_x), round(final_y)
    
    def draw_bounding_rectangle(self, surface, color=(70, 70, 70), line_width=2):
        """
        Draw the theoretical bounding rectangle of the ellipse on a Pygame surface.
        
        :param surface: Pygame surface to draw on
        :param color: RGB color of the rectangle lines (default: red)
        :param line_width: Width of the rectangle lines (default: 2)
        """
        # Calculate the four corners of the bounding rectangle
        # First, create corner points relative to the center
        corners = [
            (-self.a, -self.b),  # top-left
            (self.a, -self.b),   # top-right
            (self.a, self.b),    # bottom-right
            (-self.a, self.b)    # bottom-left
        ]
        
        # Rotate and translate the corners
        rotated_corners = []
        for x, y in corners:
            # Rotate
            x_rotated = x * math.cos(self.rotation) - y * math.sin(self.rotation)
            y_rotated = x * math.sin(self.rotation) + y * math.cos(self.rotation)
            
            # Translate
            final_x = x_rotated + self.center_x
            final_y = y_rotated + self.center_y
            
            rotated_corners.append((final_x, final_y))
        
        # Draw the rectangle lines
        pygame.draw.line(surface, color, rotated_corners[0], rotated_corners[1], line_width)
        pygame.draw.line(surface, color, rotated_corners[1], rotated_corners[2], line_width)
        pygame.draw.line(surface, color, rotated_corners[2], rotated_corners[3], line_width)
        pygame.draw.line(surface, color, rotated_corners[3], rotated_corners[0], line_width)     

class SamplingRegion:
    class QuadRegion():
        def __init__(self, quad, cumulative_area, split_ratio):
            self.quad = quad
            self.cumulative_area = cumulative_area
            self.split_ratio = split_ratio

    def __init__(self, d, k, path, nodes, mapDimensions):
        self.d_base = d
        self.d = d * k
        self.path = path
        self.maxClip = mapDimensions
        self.minClip = [0, 0]
        self.nodes = nodes
        self.expanded_region = []
        self.cache = [None] * len(self.path)
        
        self.generate_expansion_region()

    def normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def clip(self, vector):
        return np.clip(vector, self.minClip, self.maxClip)

    def direction_vector(self, v1, v2, v3):
        vec1 = self.normalize(np.array([
            self.nodes[v1].x - self.nodes[v2].x, self.nodes[v1].y - self.nodes[v2].y
        ]))
        vec2 = self.normalize(np.array([
            self.nodes[v3].x - self.nodes[v2].x, self.nodes[v3].y - self.nodes[v2].y
        ]))
        
        e_i = vec1 + vec2
        if np.linalg.norm(e_i) == 0:
            e_i = np.array([-vec1[1], vec1[0]])
        e_i = self.normalize(e_i)

        dot_product = np.dot(vec1, vec2)
        sintheta = np.sin(np.arccos(np.clip(dot_product, -1.0, 1.0))/2)

        return e_i, sintheta

    def extension(self, e_i, sintheta, i):
        v = np.array([self.nodes[i].x, self.nodes[i].y])
        e_i *= self.d / max(sintheta, 0.1)

        v_prime = self.clip(v + e_i)
        v_double_prime = self.clip(v - e_i)
        
        return [v_prime, v_double_prime]

    def extension_end(self, end, other):
        point = np.array([self.nodes[end].x, self.nodes[end].y])
        x_other, y_other = self.nodes[other].x, self.nodes[other].y
        v_to_other = self.normalize(np.array([x_other - point[0], y_other - point[1]]))

        perpendicular = np.array([-v_to_other[1], v_to_other[0]])

        v_prime_dir = -self.normalize(v_to_other + perpendicular)
        v_double_prime_dir = self.normalize(perpendicular - v_to_other)
        z = self.d/0.851

        v_prime = self.clip(point + z * v_prime_dir)
        v_double_prime = self.clip(point + z * v_double_prime_dir)

        return [v_prime, v_double_prime]
    
    def line_from_nodes(self, p1, p2):
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = a * p1.x + b * p1.y 
        return (a, b, -c)
    
    def generate_quad(self, i, cache):
        quad = cache[i] + cache[i + 1]

        eq = self.line_from_nodes(self.nodes[self.path[i]], self.nodes[self.path[i+1]])
        if self.segment_intersects_infinite_line(quad[0], quad[3], eq):
            if not self.segment_intersects_infinite_line(quad[0], quad[2], eq):
                quad[2],quad[3]=quad[3],quad[2]
            elif not self.segment_intersects_infinite_line(quad[0], quad[1], eq):
                quad[1],quad[3]=quad[3],quad[1]
        return quad

    def generate_expansion_region(self):
        self.cache[0] = self.extension_end(self.path[0], self.path[1])
        for i in range(1, len(self.path) - 1):
            e_i, sintheta = self.direction_vector(self.path[i-1], self.path[i], self.path[i+1])
            self.cache[i] = self.extension(e_i, sintheta, self.path[i])
        self.cache[-1] = self.extension_end(self.path[-1], self.path[-2])

        cumulative = 0
        for i in range(len(self.path) - 1):
            quad = self.generate_quad(i, self.cache)
            area1 = self.triangle_area(quad[0], quad[1], quad[2])
            area2 = self.triangle_area(quad[0], quad[2], quad[3])
            total = area1 + area2
            cumulative += total
            self.expanded_region.append(self.QuadRegion(
                quad=quad,
                cumulative_area=cumulative,
                split_ratio = area1 / max(total, 0.01)
            ))

    def point_line_side(self, line, p):
        return line[0] * p[0] + line[1] * p[1] + line[2]

    def segment_intersects_infinite_line(self, p1, p2, line):
        side1 = self.point_line_side(line, p1)
        side2 = self.point_line_side(line, p2)
        return side1 * side2 < 0

    def triangle_area(self, p1, p2, p3):
        return abs(np.cross(p2 - p1, p3 - p1)) / 2

    def random_point_in_triangle(self, a, b, c):
        r1 = random.random()
        r2 = random.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0])
        y = a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1])
        return round(x), round(y)

    def random_point_in_path(self):
        rand_val = random.random() * self.expanded_region[-1].cumulative_area
        left, right = 0, len(self.expanded_region) - 1
        while left < right:
            mid = (left + right) // 2
            if self.expanded_region[mid].cumulative_area < rand_val:
                left = mid + 1
            else:
                right = mid
        selected_region = left
        quad = self.expanded_region[selected_region].quad
        
        if random.random() < self.expanded_region[selected_region].split_ratio:
            return self.random_point_in_triangle(quad[0], quad[1], quad[2])
        else:
            return self.random_point_in_triangle(quad[0], quad[2], quad[3])

    def shorten_radius(self, k):
        self.d = self.d_base * k
        self.expanded_region = []
        self.generate_expansion_region()

    def draw_region(self, surface, color=(170,170,170), width=2):
        for i in range(len(self.path)-1):
            # v1,v2 = self.expanded_region[i]
            # color = (random.random()*255, random.random()*255, random.random()*255)
            # pygame.draw.circle(surface, color, v1, width+7)
            # pygame.draw.circle(surface, color, v2, width+5)
            quad = self.expanded_region[i].quad
            pygame.draw.line(surface, color, quad[0], quad[3], width)
            pygame.draw.line(surface, color, quad[1], quad[2], width)

        starts,ends = self.expanded_region[0].quad, self.expanded_region[-1].quad
        pygame.draw.line(surface, color, starts[0],starts[1], width)
        pygame.draw.line(surface, color, ends[2],ends[3], width)

class RRTGraph:
    class Node:
        def __init__(self, x, y, parent=None, children=None, cost=None):
            self.x = x
            self.y = y
            self.parent = parent
            self.children = children or set()
            self.cost = cost
        def set_parent(self, parent): self.parent = parent
        def set_cost(self, cost): self.cost = cost
        def add_child(self, child): self.children.add(child)
        def remove_child(self, child): self.children.remove(child)

    def __init__(self, start, goal, MapDimensions, surface):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.MapDimensions = MapDimensions
        self.maph, self.mapw = self.MapDimensions
        self.iter = 0

        self.num_not_in_region = 0
        self.num_max_in_region = 2000
        self.d_base = max(self.MapDimensions) / 8

        self.nodes = {0: self.Node(x, y, cost=0)}
        self.cur_index = 0
        self.kdTree = KDTree(self.nodes.items(), 2)
        
        self.goal_nodes = {0: self.Node(goal[0], goal[1], cost=0)}
        self.cur_goal_index = 0
        self.goal_kdTree = KDTree(self.goal_nodes.items(), 2)

        self.cur_tree = self.nodes

        self.best = None
        self.bestCost = float('inf')
        self.surface = surface

        self.goalstate = set()
        self.startstate = set()
        self.path = []
        self.from_goal = None

        self.sampling_region = None
        self.ellipse = None
    
    def cache_obstacle_grid(self):
        # Create the obstacle grid from the surface
        self.obstacle_grid = np.zeros((self.maph, self.mapw), dtype=bool)
        pygame_array = pygame.surfarray.array3d(self.surface)
        self.obstacle_grid = (pygame_array[:, :, 0] == 0) & (pygame_array[:, :, 1] == 0) & (pygame_array[:, :, 2] == 0)

        # Applies minimum distance that must be kept from obstacles:
        # safe_radius = 10
        # safe_zone_start = np.zeros_like(self.obstacle_grid, dtype=bool)
        # safe_zone_goal = np.zeros_like(self.obstacle_grid, dtype=bool)
        # def _mark_safe_zone(grid, position, radius):
        #     x, y = position
        #     for i in range(max(0, x - radius), min(self.mapw, x + radius + 1)):
        #         for j in range(max(0, y - radius), min(self.maph, y + radius + 1)):
        #             if np.sqrt((x - i) ** 2 + (y - j) ** 2) <= radius:
        #                 grid[i, j] = True

        # _mark_safe_zone(safe_zone_start, self.start, safe_radius)
        # _mark_safe_zone(safe_zone_goal, self.goal, safe_radius)

        # structure = np.ones((21, 21), dtype=bool)
        # dilated_grid = binary_dilation(self.obstacle_grid, structure=structure)
        # dilated_grid[safe_zone_start] = False
        # dilated_grid[safe_zone_goal] = False
        # self.obstacle_grid = dilated_grid

    def updateKDTree(self):
        self.kdTree = KDTree(self.nodes.items(), 2)
        self.goal_kdTree = KDTree(self.goal_nodes.items(), 2)

    def add_node(self, n, x, y, nodes=None):
        nodes = nodes or self.cur_tree
        from_goal = nodes == self.goal_nodes
        nodes[n] = self.Node(x,y)
        if n == self.get_cur_index(from_goal) + 1: self.get_next_index(from_goal)

    def add_edge(self, parent, child, nodes=None):
        nodes = nodes or self.cur_tree
        nodes[child].set_parent(parent)
        nodes[parent].add_child(child)

    def rewire_edge(self, parent, old_child, new_child, nodes=None):
        nodes = nodes or self.cur_tree
        nodes[parent].remove_child(old_child)
        nodes[old_child].parent = new_child
        nodes[new_child].add_child(old_child)


    def add_cost(self, node, cost, nodes=None):
        nodes = nodes or self.cur_tree
        nodes[node].set_cost(cost)

    def distance(self, n1, n2, nodes=None):
        nodes = nodes or self.cur_tree
        return math.sqrt((float(nodes[n1].x) - float(nodes[n2].x)) ** 2 + (float(nodes[n1].y) - float(nodes[n2].y)) ** 2)
    
    def calcDistance(self, x1, y1, x2, y2):
        return math.sqrt((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)
    
    def dist_points(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def estimatedCost(self, x, y):
        return self.calcDistance(x,y,self.start[0],self.start[1]) + self.calcDistance(x,y,self.goal[0],self.goal[1])

    def sample_envir(self):
        while True:
            if self.sampling_region:
                if random.random() < .87:
                    x, y = self.sampling_region.random_point_in_path()
                else:
                    x, y = self.ellipse.generate_random_point()
            else:
                x, y = random.randint(0, self.mapw - 1), random.randint(0, self.maph - 1)
            
            if not self.obstacle_grid[x, y]:
                break
        return x,y

    def cross_obstacle_points(self, startPoint, endPoint):
        line_length = self.dist_points(startPoint, endPoint)
        num_points = max(2, round(line_length / 3))
        
        points_on_line = self.get_line_points(startPoint, endPoint, num_points)
        
        for point in points_on_line:
            if self.obstacle_grid[point[0], point[1]]:
                return True
        return False

    def cross_obstacle(self, startNode, endNode, nodes=None):
        try:
            nodes = nodes or self.cur_tree
            start_pos = (nodes[startNode].x, nodes[startNode].y)
            end_pos = (nodes[endNode].x, nodes[endNode].y)
            
            return self.cross_obstacle_points(start_pos, end_pos)
        except Exception as e:
            # print(startNode, endNode, self.x[startNode], self.y[startNode], self.x[endNode], self.y[endNode], self.x, self.y)
            raise e

    def get_line_points(self, start, end, num_points):
        x1, y1 = start
        x2, y2 = end
        points = [end]
        
        if num_points < 2:
            num_points = 2

        dx = (x2 - x1) / (num_points - 1)
        dy = (y2 - y1) / (num_points - 1)

        for i in range(num_points - 1):
            x = round(x1 + i * dx)
            y = round(y1 + i * dy)
            points.append((x, y))
        
        return points
    
    def is_ancestor(self, potential_ancestor, node, nodes=None):
        nodes = nodes or self.cur_tree
        current = node
        visited = set()
        
        while current != 0:
            current = nodes[current].parent
            if current == potential_ancestor:
                return True
            if current in visited:
                break
            visited.add(current)
        
        return False

    def getCost(self, parent, child, nodes=None):
        nodes = nodes or self.cur_tree
        return nodes[parent].cost + self.distance(parent, child, nodes)

    def mergeTrees(self, node, cur_other, kdTree, nodes, other_nodes):
        cur_this = self.get_cur_index(nodes==self.goal_nodes)
        prev = node
        i = 0
        while True:
            if i==1: prev = cur_this
            cur_this += 1
            other_node = other_nodes[cur_other]
            self.add_node(cur_this, other_node.x, other_node.y, nodes)
            self.add_edge(prev, cur_this, nodes)
            nodes[cur_this].set_cost(self.getCost(prev, cur_this, nodes))
            kdTree.add_point((cur_this, other_node))
            if cur_other == 0: break
            cur_other = other_node.parent
            i = 1
        return cur_this

    def pruneTrees(self):
        self.nodes = {n: v for n, v in self.nodes.items() if len(v.children) or (self.estimatedCost(v.x,v.y) <= self.bestCost) or ((v.parent in self.nodes and n in self.nodes[v.parent].children and self.nodes[v.parent].remove_child(n) and False) or (n in self.goalstate and self.goalstate.remove(n) and False))}
        self.goal_nodes = {n: v for n, v in self.goal_nodes.items() if len(v.children) or (self.estimatedCost(v.x,v.y)) <= self.bestCost or ((v.parent in self.goal_nodes and n in self.goal_nodes[v.parent].children and self.goal_nodes[v.parent].remove_child(n) and False) or (n in self.startstate and self.startstate.remove(n) and False))}
        self.updateKDTree()

    def get_next_index(self, from_goal):
        if from_goal:
            self.cur_goal_index += 1
            return self.cur_goal_index
        else:
            self.cur_index += 1
            return self.cur_index

    def get_cur_index(self, from_goal):
        return self.cur_goal_index * from_goal + self.cur_index * (not from_goal)

    def calculate_k(self):
        z = (self.num_in_region() - self.num_max_in_region / 2)
        k = 1/(2*math.pi) * ((z<=0) * math.pi + math.atan(1 / z) if z!=0 else math.pi/2) + 0.75
        return k

    def step(self, dmax=10, bias=False, from_goal = False):
        node = self.get_cur_index(from_goal) + 1
        nodes = self.cur_tree
        other_nodes = self.nodes if from_goal else self.goal_nodes
        kdTree = self.goal_kdTree if from_goal else self.kdTree
        other_kdTree = self.kdTree if from_goal else self.goal_kdTree
        target = self.start if from_goal else self.goal
        goalstate = self.startstate if from_goal else self.goalstate
        other_goalstate = self.goalstate if from_goal else self.startstate
        point = self.sample_envir() if not bias else target

        foundGoal = bias

        if not bias and abs(point[0] - target[0]) < dmax and abs(point[1] - target[1]) < dmax:
            # jump to goal
            point = target
            foundGoal = True

        neighbors = [i[2] for i in kdTree.get_knn(point, min(10, max(5, len(nodes) // 10)), True) if i[2] != node and not (foundGoal and i[2] in goalstate) and not self.cross_obstacle_points((i[1].x, i[1].y), point)]

        # bad radial method
        # neighbors = [i for i in range(self.number_of_nodes()) if self.calcDistance(self.x[node], self.y[node], self.x[i], self.y[i]) <= dmax and i != node]
        # if len(neighbors) == 0: neighbors = [self.kdTree.get_nearest((self.x[node], self.y[node]))[2]]

        if not len(neighbors):
            return

        bestNeighbor = min(neighbors, key=lambda i: nodes[i].cost)

        self.add_node(node, point[0], point[1])
        self.add_edge(bestNeighbor, node)

        dist = self.distance(bestNeighbor, node)
        if node == self.get_cur_index(from_goal): self.add_cost(node, dist + nodes[nodes[node].parent].cost)

        for i in range(0, len(neighbors)):
            neighbor = neighbors[i]
            if neighbor == bestNeighbor: continue
            neighborCost = self.getCost(node, neighbor)
            if neighborCost < nodes[neighbor].cost and not self.is_ancestor(neighbor, node):
                nodes[neighbor].set_cost(neighborCost)

                if nodes[neighbor].cost <= 0:
                    # print(dist, dist_to_neighbor, neighbor, node, self.x[neighbor], self.y[neighbor], self.x[node], self.y[node])
                    raise Exception("Costs are negative")

                self.rewire_edge(nodes[neighbor].parent, neighbor, node)

                if nodes[nodes[neighbor].parent].cost > nodes[neighbor].cost:
                    # print(self.parent[neighbor], neighbor, self.costs[self.parent[neighbor]], self.costs[neighbor])
                    raise Exception("Parent cost is greater than the child cost")

                stack = [nodes[neighbor].children]
                while stack:
                    for child in stack.pop():
                        nodes[child].set_cost(self.getCost(nodes[child].parent, child))
                        stack.append(nodes[child].children)

        # TODO: see if this connection logic makes sense
        cur_this = node
        if not foundGoal:
            nearest_other = other_kdTree.get_nearest(point)
            nearest_other_point = (nearest_other[1].x, nearest_other[1].y)
            if nearest_other and nodes[node].cost + other_nodes[nearest_other[2]].cost < self.bestCost and self.dist_points(point, nearest_other_point) < 20  and not self.cross_obstacle_points(point, nearest_other_point):
                cur_this = self.mergeTrees(node, nearest_other[2], kdTree, nodes, other_nodes)
                other_goalstate.add(self.mergeTrees(nearest_other[2], node, other_kdTree, other_nodes, nodes))
                foundGoal = True
        # TODO: Prune trees for things above cost, figure out better metric than density to determine convergence

        kdTree.add_point((node, nodes[node]))
        if foundGoal:
            self.num_not_in_region = len(nodes)
            self.goalFlag = True
            goalstate.add(cur_this)

            old_best = self.bestCost
            self.path_to_goal()

            if self.bestCost < old_best:
                if self.bestCost <= 0:
                    raise Exception("Best path cost was negative: " + self.bestCost)
                self.sampling_region = SamplingRegion(
                    self.d_base,
                    self.calculate_k(),
                    self.path,
                    self.nodes if not self.from_goal else self.goal_nodes,
                    [self.mapw-1, self.maph-1]
                )

                self.ellipse = Ellipse(self.start,
                    self.goal,
                    self.bestCost,
                    self.mapw,
                    self.maph,
                    self.obstacle_grid
                )
                # self.sampling_region.draw_region(self.surface)
                # click = False
                # while not click:
                #     for event in pygame.event.get():
                #         if event.type == pygame.MOUSEBUTTONDOWN: click = True
            else:
                self.sampling_region.shorten_radius(self.calculate_k())

        if not foundGoal and not from_goal and self.dist_points(point, self.goal) <= 50:
            self.step(bias = True)

    def path_to_goal(self, makePath = True):
        if self.goalFlag:
            self.path = []
            
            best = min(self.goalstate, key = lambda i: self.nodes[i].cost) if len(self.goalstate) else -1
            goal_best = min(self.startstate, key = lambda i: self.goal_nodes[i].cost) if len(self.startstate) else -1
            if best == goal_best == -1: return self.goalFlag
            self.from_goal = goal_best != -1 and (best == -1 or self.goal_nodes[goal_best].cost < self.nodes[best].cost)
            self.best = goal_best if self.from_goal else best
            nodes = self.goal_nodes if self.from_goal else self.nodes
            self.bestCost = nodes[self.best].cost
            if makePath:
                self.path.append(self.best)
                newpoint = nodes[self.best].parent
                while newpoint != 0:
                    self.path.append(newpoint)
                    newpoint = nodes[newpoint].parent
                self.path.append(0)
        return self.goalFlag

    def getPathCoords(self):
        nodes = self.nodes if not self.from_goal else self.goal_nodes
        pathCoords = []
        if not self.path:
            return pathCoords
    
        cur_idx = self.path[0]
        pathCoords.append((nodes[cur_idx].x, nodes[cur_idx].y))
        
        i = 0
        cur_node = self.path[0]
        while i < len(self.path)-1:
            cur = self.path[i]
            next_node = self.path[i + 1]
            
            if self.cross_obstacle(cur_node, next_node, nodes):
                pathCoords.append((nodes[cur].x, nodes[cur].y))
                cur_node = cur
            i += 1
        
        pathCoords.append((nodes[self.path[-1]].x, nodes[self.path[-1]].y))
        return pathCoords

    def bias(self):
        self.step(bias=True)
        return self.nodes, self.goal_nodes

    def expand(self):
        if self.iter % 2 == 0:
            self.iter = 0
            self.cur_tree = self.nodes
            self.step()
        else:
            self.cur_tree = self.goal_nodes
            self.step(from_goal=True)
        self.iter += 1
        return self.nodes, self.goal_nodes

    def num_in_region(self):
        return len(self.nodes)+len(self.goal_nodes) - self.num_not_in_region if self.num_not_in_region is not None else 0


def waitClick():
    click = False
    while not click:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN: click = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE : click = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o : return True

def main():
    dimensions = (600, 1000)
    start = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
    goal = (random.randint(0, dimensions[1]- 1), random.randint(50, dimensions[0] - 1))

    #close to edge:
    # start = (0,0)
    # goal = (300,300)

    # multiple potential paths:
    # start = (40, 13) 
    # goal = (910, 283) 

    # MAZE
    # start = (50, 50)
    # goal = (950, 550)

    obsdim = 60
    obsnum = 50
    iteration = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    graph = RRTGraph(start, goal, dimensions, map.map)

    obstacles = map.makeobs()
    # obstacles=[]
    # <rect\((\d*), (\d*), (\d*), (\d*)\)> ($1, $2, $3, $4)
    # close to edge:
    # obstacles = [('rect', (50, 50, 50, 100)), ('rect', (200, 50, 50, 490)), ('rect', (200, 290, 300, 10)), ('rect', (310, 310, 50, 1000))]

    # multiple potential paths:
    # obstacles = [('triangle', [(200, 437), (251, 437), (218, 379)]), ('circle', ((962, 125), 26)), ('rect', (479, 212, 38, 47)), ('circle', ((857, 314), 19)), ('circle', ((564, 340), 19)), ('triangle', [(501, 112), (558, 112), (524, 68)]), ('circle', ((521, 535), 15)), ('rect', (883, 472, 46, 32)), ('triangle', [(221, 220), (279, 220), (251, 180)]), ('triangle', [(885, 417), (937, 417), (909, 386)]), ('circle', ((208, 330), 29)), ('triangle', [(937, 32), (973, 32), (965, -1)]), ('triangle', [(468, 455), (527, 455), (483, 412)]), ('circle', ((803, 92), 18)), ('triangle', [(673, 236), (731, 236), (694, 201)]), ('circle', ((840, 166), 28)), ('circle', ((134, 65), 26)), ('triangle', [(818, 516), (855, 516), (837, 467)]), ('rect', (516, 417, 53, 41)), ('triangle', [(693, 375), (732, 375), (721, 324)]), ('circle', ((256, 443), 29)), ('circle', ((914, 96), 17)), ('triangle', [(497, 400), (554, 400), (515, 340)]), ('rect', (884, 31, 45, 41)), ('triangle', [(152, 90), (201, 90), (181, 43)]), ('circle', ((898, 403), 23)), ('circle', ((318, 198), 28)), ('triangle', [(226, 406), (260, 406), (255, 349)]), ('rect', (819, 103, 47, 57)), ('rect', (437, 105, 45, 45)), ('triangle', [(506, 454), (550, 454), (529, 399)]), ('circle', ((108, 237), 20)), ('triangle', [(769, 381), (819, 381), (795, 351)]), ('triangle', [(825, 413), (884, 413), (841, 377)]), ('circle', ((236, 205), 19)), ('rect', (245, 307, 56, 34)), ('rect', (595, 376, 52, 33)), ('triangle', [(49, 451), (84, 451), (65, 396)]), ('triangle', [(760, 6), (808, 6), (782, -36)]), ('circle', ((766, 335), 25)), ('circle', ((322, 298), 28)), ('rect', (478, 158, 37, 58)), ('rect', (689, 55, 50, 41)), ('circle', ((608, 395), 27)), ('circle', ((169, 200), 16)), ('circle', ((226, 303), 17)), ('triangle', [(133, 96), (163, 96), (155, 48)]), ('circle', ((607, 239), 17)), ('triangle', [(749, 425), (798, 425), (765, 366)]), ('rect', (54, 223, 48, 51))]
    
    # MAZE
    # obstacles = [
    #     # Outer walls with a small entry and exit point
    #     ('rect', (0, 0, 1000, 10)),  # Top wall
    #     ('rect', (0, 0, 10, 600)),   # Left wall
    #     ('rect', (0, 590, 1000, 10)),  # Bottom wall
    #     ('rect', (990, 0, 10, 600)),  # Right wall

    #     # Vertical barriers
    #     ('rect', (100, 50, 10, 500)),
    #     ('rect', (300, 0, 10, 450)),
    #     ('rect', (500, 150, 10, 420)),
    #     ('rect', (700, 20, 10, 450)),
    #     ('rect', (900, 150, 10, 450)),

    #     # Horizontal barriers
    #     ('rect', (10, 200, 270, 10)),
    #     ('rect', (310, 400, 190, 10)),
    #     ('rect', (510, 100, 190, 10)),
    #     ('rect', (710, 300, 170, 10)),
    #     ('rect', (850, 200, 50, 10)),
    #     ('rect', (740, 100, 100, 10)),
        
    #     # Central labyrinth area
    #     ('rect', (400, 200, 10, 100)),
    #     ('rect', (450, 250, 50, 10)),
    #     ('rect', (500, 200, 10, 100)),
    #     ('rect', (450, 200, 50, 10)),

    #     # Additional tricky paths
    #     ('rect', (150, 300, 10, 100)),
    #     ('rect', (200, 250, 10, 100)),
    #     ('rect', (250, 300, 10, 100)),
    #     ('rect', (350, 150, 100, 10)),
    #     ('rect', (550, 350, 100, 10)),
    #     ('rect', (850, 100, 10, 100)),
    #     ('rect', (850, 400, 10, 100)),

    #     # Dead ends
    #     ('rect', (100, 550, 50, 10)),
    #     ('rect', (400, 550, 50, 10)),
    #     ('rect', (750, 550, 50, 10)),

    #     # circles
    #     # ('circle', ((800, 250), 50, 50)),
    #     # ('circle', ((400, 100), 50, 50)),
    # ]

    map.drawMap(obstacles)

    pygame.display.update()
    pygame.event.clear()

    if waitClick(): print(obstacles, start, goal)
    
    distToGoal = math.dist(start, goal)
    start_time = pygame.time.get_ticks()
    font = pygame.font.Font(None, 36)

    isSolved = False
    graph.cache_obstacle_grid()
    graph.bias()

    def updateDisplay(nodes, goal_nodes, isSolved):
        map.map.fill((255,255,255))
        map.drawMap(obstacles)
        timer_text = font.render(f"Time: {elapsed_time:.2f} s", True, (0, 0, 0))
        map.map.blit(timer_text, (10, 10))
        
        for v in nodes.values():
            pygame.draw.circle(map.map, map.grey if (v.x != goal[0] or v.y != goal[1]) and (v.x != start[0] or v.y != start[1]) else map.Red, (v.x, v.y), map.nodeRad+1)
            if v.parent is not None:
                pygame.draw.line(map.map, map.Blue, (v.x, v.y), (nodes[v.parent].x, nodes[v.parent].y), map.edgeThickness)
        for v in goal_nodes.values():
            pygame.draw.circle(map.map, map.grey, (v.x, v.y), map.nodeRad+1)
            if v.parent is not None:
                pygame.draw.line(map.map, map.Green, (v.x, v.y), (goal_nodes[v.parent].x, goal_nodes[v.parent].y), map.edgeThickness)
        if isSolved:
            graph.sampling_region.draw_region(map.map)
            nodes = graph.goal_nodes if graph.from_goal else graph.nodes
            graph.path_to_goal()
            map.drawPath(graph.getPathCoords(), graph.from_goal)
            pygame.display.update()
            if abs(graph.bestCost - distToGoal) < 1:
                return True
    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - start_time) / 1000.0
        # print(min(max(graph.bestCost / 2000 * 75 + 275, 275), 350))
        if graph.bestCost and ((graph.num_in_region()) / math.log(graph.bestCost + 2) > min(max(graph.bestCost / 2000 * 75 + 275, 275), 350)):
            break
            area = graph.get_region_area()
            if area and ((graph.num_in_region()) / max(math.log(area + 2), 0.8) > 290):
                break
        # if iteration % 1000 == 0:
        #     X, Y, Parent = graph.bias()
        #     pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad, map.nodeRad + 4, 0)
        #     pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        # else:
        nodes, goal_nodes = graph.expand()
        
        #     pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad+4)
        #     pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)

        
        if not isSolved: 
            isSolved = graph.path_to_goal(False)
            if iteration % 500 == 0: graph.updateKDTree()
        if iteration % 1000 == 0:
            if isSolved and graph.num_in_region() / graph.bestCost < 2:
                graph.pruneTrees()
            if updateDisplay(nodes, goal_nodes, isSolved): break
            # waitClick()
            graph.path = []

        iteration += 1
    
    current_time = pygame.time.get_ticks()
    elapsed_time = (current_time - start_time) / 1000.0
    updateDisplay(nodes, goal_nodes, True)
    timer_text = font.render(f"Total Time: {elapsed_time:.2f} s", True, (16, 150, 34))
    map.map.blit(timer_text, (10, 30))
    pygame.display.update()
    pygame.event.clear()
    print(graph.bestCost)
    if waitClick(): print(obstacles, start, goal)
    main()


if __name__ == "__main__":
    main()

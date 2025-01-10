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
        for i in range(len(path)):
            pygame.draw.circle(self.map, self.Red, path[i], self.nodeRad + 4, 0)
            if i < len(path) - 1:
                pygame.draw.line(self.map, self.Red if not from_goal else (250,95,85), path[i], path[i + 1], self.nodeRad +  3)

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
    def __init__(self, d, k, path, nodes, mapDimensions):
        self.d_base = d
        self.d = d * k
        self.path = path
        self.maxClip = mapDimensions
        self.minClip = [0, 0]
        self.nodes = nodes
        self.expanded_region = {}
        self.region_quads = []
        self.region_areas = []
        self.triangle_splits = []  
        
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
        e_i *= self.d / sintheta

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

    def generate_expansion_region(self):
        for i in range(1, len(self.path) - 1):
            e_i, sintheta = self.direction_vector(self.path[i-1], self.path[i], self.path[i+1])
            self.expanded_region[i] = self.extension(e_i, sintheta, self.path[i])
        self.expanded_region[0] = self.extension_end(self.path[0], self.path[1])
        self.expanded_region[len(self.path)-1] = self.extension_end(self.path[-1], self.path[-2])
        
        self.precompute_region_data()

    def sign(self, p1, center, p2):
        return (p1[0]-center[0])*(p2[1]-center[1])-(p1[1]-center[1])*(p2[0]-center[0]) <= 0

    def precompute_region_data(self):
        total_area = 0
        
        for i in range(len(self.path)-1):
            c1,c2 = self.expanded_region[i]
            n1,n2 = self.expanded_region[i+1]

            if self.sign(c2,c1,n2) != self.sign(n2,n1,c2):
                n1,n2=n2,n1

            elif self.sign(c2,c1,n1) != self.sign(n2,c1,c2):
                c1,n2=n2,c1

            quad = (
                c1,
                c2,
                n1,
                n2
            )
            self.region_quads.append(quad)
            
            area1 = self.triangle_area(*quad[0], *quad[1], *quad[2])
            area2 = self.triangle_area(*quad[0], *quad[2], *quad[3])
            
            total_area += (area1 + area2)
            self.region_areas.append(total_area)
            self.triangle_splits.append(area1 / (area1 + area2))  

    def triangle_area(self, x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

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
        rand_val = random.random() * self.region_areas[-1]
        left, right = 0, len(self.region_areas) - 1
        while left < right:
            mid = (left + right) // 2
            if self.region_areas[mid] < rand_val:
                left = mid + 1
            else:
                right = mid
        selected_region = left
        quad = self.region_quads[selected_region]
        
        if random.random() < self.triangle_splits[selected_region]:
            return self.random_point_in_triangle(quad[0], quad[1], quad[2])
        else:
            return self.random_point_in_triangle(quad[0], quad[2], quad[3])

    def shorten_radius(self, k):
        self.d = self.d_base * k
        self.expanded_region = {}
        self.region_quads = []
        self.region_areas = []
        self.triangle_splits = []
        self.generate_expansion_region()

    def draw_region(self, surface, color=(255, 0, 255), width=4):
        # for key in range(len(self.path)):
        #     vertices = self.expanded_region[key]
            # pygame.draw.line(surface, color, vertices[0], vertices[1], width)

        for i in range(len(self.path)-1):
            # v1 = self.expanded_region[i][0]
            # v2 = self.expanded_region[i + 1][0]
            # v3 = self.expanded_region[i][1]
            # v4 = self.expanded_region[i + 1][1]

            # pygame.draw.line(surface, color, v1, v2, width)
            # pygame.draw.line(surface, color, v3, v4, width)

            v1,v2 = self.expanded_region[i]
            v3,v4 = self.expanded_region[i+1]
            color = (random.random()*255, random.random()*255, random.random()*255)

            # pygame.draw.circle(surface, color, v1, width+5)
            # pygame.draw.circle(surface, color, v2, width+5)
            pygame.draw.circle(surface, color, v3, width+5)
            pygame.draw.circle(surface, color, v4, width+5)
            quad = self.region_quads[i]
            pygame.draw.line(surface, color, quad[0], quad[3], width)
            pygame.draw.line(surface, color, quad[1], quad[2], width)

        starts,ends = self.region_quads[0], self.region_quads[-1]
        pygame.draw.line(surface, color, starts[0],starts[1])
        pygame.draw.line(surface, color, ends[2],ends[3])

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
        self.obstacle_grid = np.zeros((self.maph, self.mapw), dtype=bool)
        pygame_array = pygame.surfarray.array3d(self.surface)
        self.obstacle_grid = (pygame_array[:, :, 0] == 0) & (pygame_array[:, :, 1] == 0) & (pygame_array[:, :, 2] == 0)

        # Applies minimum distance that must be kept from obstacles:
        # structure = np.ones((21, 21), dtype=bool) # radius 10
        # self.obstacle_grid = binary_dilation(self.obstacle_grid, structure=structure)

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
            print(startNode, endNode, self.x[startNode], self.y[startNode], self.x[endNode], self.y[endNode], self.x, self.y)
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
        for i in self.path:
            pathCoords.append((nodes[i].x, nodes[i].y))
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

    # start, goal = (750,150), (500,160)
    # start, goal = (700, 520), (90, 90)

    obsdim = 60
    obsnum = 50
    iteration = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    graph = RRTGraph(start, goal, dimensions, map.map)

    obstacles = map.makeobs()
    # <rect\((\d*), (\d*), (\d*), (\d*)\)> ($1, $2, $3, $4)
    # confined = [('triangle', [(746, 479), (798, 479), (769, 437)]), ('circle', ((354, 448), 25)), ('rect', (386, 141, 56, 46)), ('circle', ((809, 116), 21)), ('rect', (885, 301, 47, 37)), ('rect', (621, 400, 41, 45)), ('rect', (859, 487, 53, 38)), ('circle', ((714, 188), 30)), ('rect', (638, 230, 31, 56)), ('rect', (413, 256, 50, 54)), ('circle', ((616, 423), 15)), ('circle', ((828, 244), 19)), ('circle', ((497, 314), 20)), ('circle', ((597, 393), 23)), ('circle', ((836, 81), 18)), ('circle', ((97, 95), 22)), ('rect', (724, 32, 44, 49)), ('triangle', [(340, 118), (370, 118), (369, 63)]), ('rect', (90, 535, 40, 52)), ('circle', ((449, 122), 17)), ('rect', (245, 235, 56, 43)), ('triangle', [(969, 470), (1027, 470), (991, 431)]), ('circle', ((185, 405), 15)), ('rect', (70, 561, 47, 53)), ('circle', ((626, 423), 26)), ('circle', ((153, 267), 22)), ('triangle', [(103, 106), (159, 106), (119, 63)]), ('triangle', [(359, 193), (395, 193), (387, 159)]), ('circle', ((83, 504), 19)), ('triangle', [(481, 373), (540, 373), (504, 334)]), ('triangle', [(262, 97), (297, 97), (278, 64)]), ('triangle', [(285, 118), (327, 118), (309, 85)]), ('triangle', [(321, 185), (359, 185), (345, 155)]), ('circle', ((924, 203), 19)), ('triangle', [(403, 261), (440, 261), (427, 207)]), ('triangle', [(828, 234), (865, 234), (855, 202)]), ('rect', (477, 556, 60, 44)), ('rect', (310, 16, 53, 56)), ('rect', (752, 94, 45, 55)), ('rect', (364, 78, 35, 47)), ('circle', ((678, 143), 26)), ('triangle', [(751, 36), (796, 36), (766, -15)]), ('triangle', [(524, 277), (558, 277), (540, 222)]), ('circle', ((224, 257), 22)), ('rect', (462, 492, 31, 46)), ('triangle', [(16, 318), (60, 318), (41, 276)]), ('circle', ((790, 416), 24)), ('rect', (688, 176, 33, 50)), ('rect', (315, 444, 35, 31)), ('circle', ((742, 469), 24))]
    # confined = [('triangle', [(389, 455), (431, 455), (406, 412)]), ('circle', ((188, 200), 25)), ('triangle', [(572, 114), (613, 114), (597, 56)]), ('triangle', [(123, 365), (172, 365), (151, 315)]), ('triangle', [(305, 413), (347, 413), (327, 361)]), ('circle', ((375, 320), 28)), ('rect', (425, 203, 58, 48)), ('rect', (176, 85, 34, 49)), ('triangle', [(32, 297), (72, 297), (61, 265)]), ('circle', ((662, 421), 20)), ('triangle', [(879, 494), (914, 494), (901, 439)]), ('triangle', [(842, 316), (884, 316), (862, 257)]), ('circle', ((586, 447), 27)), ('triangle', [(955, 2), (1013, 2), (975, -46)]), ('circle', ((947, 60), 28)), ('triangle', [(241, 206), (280, 206), (269, 154)]), ('triangle', [(378, 257), (438, 257), (400, 199)]), ('circle', ((660, 422), 26)), ('rect', (533, 491, 57, 39)), ('circle', ((929, 368), 26)), ('rect', (745, 516, 48, 42)), ('rect', (501, 440, 51, 34)), ('rect', (357, 419, 40, 39)), ('triangle', [(862, 538), (918, 538), (886, 506)]), ('circle', ((837, 404), 30)), ('triangle', [(694, 346), (738, 346), (723, 292)]), ('triangle', [(769, 452), (826, 452), (790, 402)]), ('triangle', [(891, 357), (938, 357), (918, 308)]), ('rect', (961, 273, 35, 50)), ('circle', ((157, 458), 19)), ('circle', ((483, 436), 28)), ('rect', (935, 232, 46, 49)), ('rect', (850, 79, 50, 30)), ('triangle', [(776, 124), (830, 124), (798, 69)]), ('rect', (136, 189, 39, 48)), ('rect', (377, 257, 32, 46)), ('triangle', [(195, 439), (252, 439), (222, 385)]), ('triangle', [(585, 76), (640, 76), (608, 43)]), ('circle', ((574, 440), 17)), ('rect', (327, 333, 51, 58)), ('triangle', [(911, 468), (945, 468), (926, 424)]), ('triangle', [(782, 190), (826, 190), (801, 155)]), ('rect', (961, 88, 38, 55)), ('circle', ((858, 427), 16)), ('circle', ((964, 335), 29)), ('circle', ((466, 278), 26)), ('triangle', [(631, 284), (683, 284), (647, 248)]), ('rect', (19, 237, 55, 46)), ('triangle', [(443, 337), (496, 337), (459, 307)]), ('triangle', [(739, 183), (771, 183), (768, 150)])]
    # obstacles = confined
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
            # graph.ellipse.draw_bounding_rectangle(map.map)
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
        if graph.bestCost and ((graph.num_in_region()) / math.log(graph.bestCost + 2) > 275):
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

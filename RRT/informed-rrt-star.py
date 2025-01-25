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
    def __init__(self, start, goal, MapDimensions):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.maph, self.mapw = self.MapDimensions

        self.MapWindowName = "Informed RRT*"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.fill((255, 255, 255))
        self.nodeRad = 0
        self.nodeThickness = 0
        self.edgeThickness = 1

        self.black = (0, 0, 0)
        self.grey = (75, 75, 75)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)

    def drawMap(self, obstacles):
        pygame.draw.circle(self.map, self.Red, self.start, self.nodeRad + 5, 0)
        pygame.draw.circle(self.map, self.Green, self.goal, self.nodeRad + 5, 0)
        obstacles and self.drawObs(obstacles)

    def drawPath(self, path, from_goal = False):
        for i in range(len(path)):
            # pygame.draw.circle(self.map, self.Red, path[i], self.nodeRad + 4, 0)
            if i < len(path) - 1:
                pygame.draw.line(self.map, self.Red if not from_goal else (250,95,85), path[i], path[i + 1], self.nodeRad +  4)

    def resetPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.Blue, node, self.nodeRad+4, 0)

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
        self.num_not_in_ellipse = 0
        self.MapDimensions = MapDimensions
        self.maph, self.mapw = self.MapDimensions
        self.iter = 0

        self.nodes = {0: self.Node(x, y, cost=0)}
        self.cur_index = 0
        self.kdTree = KDTree(self.nodes.items(), 2)
        
        self.goal_nodes = {0: self.Node(goal[0], goal[1], cost=0)}
        self.cur_goal_index = 0
        self.goal_kdTree = KDTree(self.goal_nodes.items(), 2)

        self.cur_tree = self.nodes

        self.best = None
        self.bestCost = float('inf')
        # self.rTree = RTree()
        # self.rTree.insert((x,y), 0)
        self.surface = surface

        self.goalstate = set()
        self.startstate = set()
        self.path = []
        self.from_goal = None
        self.ellipse = None
    
    def cache_obstacle_grid(self):
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
            if self.ellipse:
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
            self.num_not_in_ellipse = len(nodes)
            self.goalFlag = True
            goalstate.add(cur_this)

            old_best = self.bestCost
            self.path_to_goal(False)            

            if self.bestCost < old_best:
                if self.bestCost <= 0:
                    raise Exception("Best path cost was negative: " + self.bestCost)

                self.ellipse = Ellipse(
                    self.start,
                    self.goal,
                    self.bestCost,
                    self.mapw,
                    self.maph,
                    self.obstacle_grid
                )

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
            
        self.path
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

    def num_in_ellipse(self):
        return len(self.nodes)+len(self.goal_nodes) - self.num_not_in_ellipse if self.num_not_in_ellipse is not None else 0


def waitClick():
    click = False
    while not click:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN: click = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE : click = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o : return True

def makeRandomShape(obsDim, mapw, maph):
    shape_type = random.choice(["rect", "circle", "triangle"])

    if shape_type == "rect":
        x = random.randint(0, mapw - 30)
        y = random.randint(0, maph - 30)
        return "rect", pygame.Rect((x, y), (random.randint(30, obsDim), random.randint(30, obsDim)))

    elif shape_type == "circle":
        x = random.randint(obsDim, mapw - 30)
        y = random.randint(obsDim, maph - 30)
        radius = random.randint(30, obsDim) // 2
        return "circle", ((x, y), radius)

    elif shape_type == "triangle":
        x1 = random.randint(0, mapw - 30)
        y1 = random.randint(0, maph - 30)
        x2 = x1 + random.randint(30, obsDim)
        y2 = y1
        x3 = x1 + random.randint(30, obsDim) // 2
        y3 = y1 - random.randint(30, obsDim)
        return "triangle", [(x1, y1), (x2, y2), (x3, y3)]

def makeobs(obsDim, obsNum, start, goal, mapw, maph):
    obs = []
    goal_circle = pygame.Rect(goal[0] - 5, goal[1] - 5, 10, 10)
    start_circle = pygame.Rect(start[0] - 5, start[1] - 5, 10, 10)

    for _ in range(obsNum):
        shape = None
        startGoalCol = True
        while startGoalCol:
            shape = makeRandomShape(obsDim, mapw, maph)
            shape_type, properties = shape
            if shape_type == "rect":
                rect = properties
                startGoalCol = rect.colliderect(goal_circle) or rect.colliderect(start_circle)
            elif shape_type == "circle":
                circle_center, radius = properties
                circle_rect = pygame.Rect(circle_center[0] - radius, circle_center[1] - radius, 2 * radius, 2 * radius)
                startGoalCol = circle_rect.colliderect(goal_circle) or circle_rect.colliderect(start_circle)
            elif shape_type == "triangle":
                triangle_rect = pygame.Rect(min(p[0] for p in properties), min(p[1] for p in properties),
                                            obsDim, obsDim)
                startGoalCol = triangle_rect.colliderect(goal_circle) or triangle_rect.colliderect(start_circle)

        obs.append(shape)
    return obs


def main():
    dimensions = (600, 1000)
    obsdim = 60
    obsnum = 50
    start, goal, obstacles, image = None, None, None, None
    
    pygame.init()

    # Options: random, maze, narrow, close to edge
    config = "random"

    if config == "random":
        start = (random.randint(0, dimensions[1] - 1), random.randint(0, dimensions[0] - 1))
        goal = (random.randint(0, dimensions[1]- 1), random.randint(50, dimensions[0] - 1))
        obstacles = makeobs(obsdim, obsnum, start, goal, dimensions[1], dimensions[0])
    elif config == "maze":
        start = (50, 50)
        goal = (950, 550)
        obstacles = [
            # Outer walls with a small entry and exit point
            ('rect', (0, 0, 1000, 10)),  # Top wall
            ('rect', (0, 0, 10, 600)),   # Left wall
            ('rect', (0, 590, 1000, 10)),  # Bottom wall
            ('rect', (990, 0, 10, 600)),  # Right wall

            # Vertical barriers
            ('rect', (100, 50, 10, 500)),
            ('rect', (300, 0, 10, 450)),
            ('rect', (500, 150, 10, 420)),
            ('rect', (700, 20, 10, 450)),
            ('rect', (900, 150, 10, 450)),

            # Horizontal barriers
            ('rect', (10, 200, 270, 10)),
            ('rect', (310, 400, 190, 10)),
            ('rect', (510, 100, 190, 10)),
            ('rect', (710, 300, 170, 10)),
            ('rect', (850, 200, 50, 10)),
            ('rect', (740, 100, 100, 10)),
            
            # Central labyrinth area
            ('rect', (400, 200, 10, 100)),
            ('rect', (450, 250, 50, 10)),
            ('rect', (500, 200, 10, 100)),
            ('rect', (450, 200, 50, 10)),

            # Additional tricky paths
            ('rect', (150, 300, 10, 100)),
            ('rect', (200, 250, 10, 100)),
            ('rect', (250, 300, 10, 100)),
            ('rect', (350, 150, 100, 10)),
            ('rect', (550, 350, 100, 10)),
            ('rect', (850, 100, 10, 100)),
            ('rect', (850, 400, 10, 100)),

            # Dead ends
            ('rect', (100, 550, 50, 10)),
            ('rect', (400, 550, 50, 10)),
            ('rect', (750, 550, 50, 10)),

            # circles
            # ('circle', ((800, 250), 50, 50)),
            # ('circle', ((400, 100), 50, 50)),
        ]
    elif config == "close to edge":
        start = (0,0)
        goal = (300,300)
        obstacles = [('rect', (50, 50, 50, 100)), ('rect', (200, 50, 50, 490)), ('rect', (200, 290, 300, 10)), ('rect', (310, 310, 50, 1000))]
    elif config == "narrow":
        start = (50, 50)
        goal = (950, 550)
        try:
            map = RRTMap(start, goal, dimensions)
            graph = RRTGraph(start, goal, dimensions, map.map)
            image = pygame.image.load(r'../maps/narrow-corridor.jpg')
            pygame.display.get_surface().blit(image, (0,0))
        except Exception as e:
            print("Error reading file:", e)
            return
    else:
        print("Invalid configuration!")
        return

    if config not in ['narrow']:
        map = RRTMap(start, goal, dimensions)
        graph = RRTGraph(start, goal, dimensions, map.map)

    map.drawMap(obstacles)

    pygame.display.update()
    pygame.event.clear()

    if waitClick():
        print("Obstacles:", obstacles, "Start:", start, "Goal:", goal)
    
    distToGoal = math.dist(start, goal)
    start_time = pygame.time.get_ticks()
    font = pygame.font.Font(None, 36)

    isSolved = False
    graph.cache_obstacle_grid()
    graph.bias()
    iteration = 0

    def updateDisplay(nodes, goal_nodes, isSolved):
        map.map.fill((255,255,255))
        map.drawMap(obstacles)
        image and pygame.display.get_surface().blit(image, (0,0))
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
            graph.ellipse.draw_bounding_rectangle(map.map)
            nodes = graph.goal_nodes if graph.from_goal else graph.nodes
            if abs(nodes[graph.best].cost - distToGoal) < 1:
                pygame.display.update()
                return True
            graph.path_to_goal()
            map.drawPath(graph.getPathCoords(), graph.from_goal)
            pygame.display.update()
    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - start_time) / 1000.0
        if graph.num_not_in_ellipse and graph.ellipse:
            area = graph.ellipse.get_area()
            if area and ((graph.num_in_ellipse()) / max(math.log(area + 2), 0.8) > 290):
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
            if isSolved and graph.num_in_ellipse() / graph.bestCost < 2:
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
    if waitClick(): print(obstacles, start, goal)
    main()


if __name__ == "__main__":
    main()

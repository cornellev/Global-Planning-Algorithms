import random
import math
import pygame
import numpy as np
import heapq

class KDTree(object):
    def __init__(self, points, dim, dist_sq_func=None):
        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 
                for i, x in enumerate(a))
                
        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[0][i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), 
                    points[m]]
            if len(points) == 1:
                return [None, None, points[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][0][i] - point[0][i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2][0]) if not (node[2][0][0] == point[0] and node[2][0][1] == point[1]) else float('inf')
                dx = node[2][0][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2][0], node[2][1]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2][0], node[2][1]))
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

    def drawPath(self, path):
        for i in range(len(path)):
            pygame.draw.circle(self.map, self.Red, path[i], self.nodeRad + 4, 0)
            if i < len(path) - 1:
                pygame.draw.line(self.map, self.Red, path[i], path[i + 1], self.nodeRad + 3)

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

class RRTGraph:
    def __init__(self, start, goal, MapDimensions, surface):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.num_not_in_ellipse = 0
        self.MapDimensions = MapDimensions
        self.maph, self.mapw = self.MapDimensions

        self.x = [x]
        self.y = [y]
        self.parent = [0]
        self.children = [set()]
        self.costs = [0]
        self.kdTree = KDTree([((x, y), 0)], 2)
        self.best = None
        # self.rTree = RTree()
        # self.rTree.insert((x,y), 0)
        self.surface = surface

        self.goalstate = set()
        self.path = []
        self.ellipse = None
    
    def cache_obstacle_grid(self):
        self.obstacle_grid = np.zeros((self.maph, self.mapw), dtype=bool)
        pygame_array = pygame.surfarray.array3d(self.surface)
        self.obstacle_grid = (pygame_array[:, :, 0] == 0) & (pygame_array[:, :, 1] == 0) & (pygame_array[:, :, 2] == 0)

    def updateKDTree(self):
        self.kdTree = KDTree([((self.x[i], self.y[i]), i) for i in range(self.number_of_nodes())], 2)

    def add_node(self, n, x, y):
        self.x.insert(n,x)
        self.y.insert(n,y)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent)
        self.children[parent].add(child)
        self.children.insert(child, set())

    def add_cost(self, node, cost):
        self.costs.insert(node, cost)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self, n1, n2):
        return math.sqrt((float(self.x[n1]) - float(self.x[n2])) ** 2 + (float(self.y[n1]) - float(self.y[n2])) ** 2)
    
    def calcDistance(self, x1, y1, x2, y2):
        return math.sqrt((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)

    def sample_envir(self):
        while True:
            if self.ellipse:
                x, y = self.ellipse.generate_random_point()
            else:
                x, y = random.randint(0, self.mapw - 1), random.randint(0, self.maph - 1)
            
            if not self.obstacle_grid[x, y]:
                break
        return x,y

    def dist_points(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def cross_obstacle_points(self, startPoint, endPoint):
        line_length = self.dist_points(startPoint, endPoint)
        num_points = max(2, round(line_length / 3))
        
        points_on_line = self.get_line_points(startPoint, endPoint, num_points)
        
        for point in points_on_line:
            if self.obstacle_grid[point[0], point[1]]:
                return True
        return False

    def cross_obstacle(self, startNode, endNode):
        try:
            start_pos = (self.x[startNode], self.y[startNode])
            end_pos = (self.x[endNode], self.y[endNode])
            
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
    
    def is_ancestor(self, potential_ancestor, node):
        current = node
        visited = set()
        
        while current != 0:
            current = self.parent[current]
            if current == potential_ancestor:
                return True
            if current in visited:
                break
            visited.add(current)
        
        return False

    def step(self, node, dmax=10, bias=False):

        point = self.sample_envir() if not bias else self.goal

        foundGoal = bias

        if not bias and abs(point[0] - self.goal[0]) < dmax and abs(point[1] - self.goal[1]) < dmax:
            # jump to goal
            point = self.goal
            foundGoal = True

        neighbors = [i[2] for i in self.kdTree.get_knn(point, min(10, max(5, len(self.x) // 10)), True) if i[2] != node and not (foundGoal and i[2] in self.goalstate) and not self.cross_obstacle_points(i[1], point)]

        # bad radial method
        # neighbors = [i for i in range(self.number_of_nodes()) if self.calcDistance(self.x[node], self.y[node], self.x[i], self.y[i]) <= dmax and i != node]
        # if len(neighbors) == 0: neighbors = [self.kdTree.get_nearest((self.x[node], self.y[node]))[2]]

        if not len(neighbors):
            return

        bestNeighbor = min(neighbors, key=lambda i: self.costs[i])

        self.add_node(node, point[0], point[1])
        self.add_edge(bestNeighbor, node)
        
        dist = self.distance(bestNeighbor, node)
        if node == self.number_of_nodes()-1: self.add_cost(node, dist + self.costs[self.parent[node]])

        if foundGoal:
            self.goalstate.add(node)
            if not self.goalFlag: self.num_not_in_ellipse = self.number_of_nodes()
            self.goalFlag = foundGoal = True
            
            self.path_to_goal()
            pathLength = self.costs[self.best]

            if (not self.ellipse) or (pathLength < self.ellipse.a * 2):
                self.ellipse = Ellipse(
                    self.start,
                    self.goal,
                    pathLength,
                    self.mapw,
                    self.maph,
                    self.obstacle_grid
                )

        for i in range(0, len(neighbors)):
            neighbor = neighbors[i]
            if neighbor == bestNeighbor: continue
            dist_to_neighbor = self.distance(neighbor, node)
            neighborCost = self.costs[node] + dist_to_neighbor
            if neighborCost < self.costs[neighbor] and not self.is_ancestor(neighbor, node):
                self.costs[neighbor] = neighborCost

                if self.costs[neighbor] <= 0:
                    # print(dist, dist_to_neighbor, neighbor, node, self.x[neighbor], self.y[neighbor], self.x[node], self.y[node])
                    raise Exception("Costs are negative")

                self.children[self.parent[neighbor]].remove(neighbor)
                self.parent[neighbor] = node
                self.children[node].add(neighbor)

                if self.costs[self.parent[neighbor]] > self.costs[neighbor]:
                    # print(self.parent[neighbor], neighbor, self.costs[self.parent[neighbor]], self.costs[neighbor])
                    raise Exception("Parent cost is greater than the child cost")

                stack = [self.children[neighbor]]
                while stack:
                    for child in stack.pop():
                        self.costs[child] = self.costs[self.parent[child]] + self.distance(self.parent[child], child)
                        stack.append(self.children[child])

        self.kdTree.add_point((point, node))

        if not foundGoal and self.dist_points(point, self.goal) <= 50:
            self.step(self.number_of_nodes(), bias = True)

    def path_to_goal(self):
        if self.goalFlag:
            self.path = []
            self.best = min(self.goalstate, key = lambda i: self.costs[i])
            self.path.append(self.best)
            newpoint = self.parent[self.best]
            while newpoint != 0:
                self.path.append(newpoint)
                newpoint = self.parent[newpoint]
            self.path.append(0)
        return self.goalFlag

    def getPathCoords(self):
        pathCoords = []
        for i in self.path:
            pathCoords.append((self.x[i], self.y[i]))
        return pathCoords

    def bias(self):
        n = self.number_of_nodes()
        self.step(n, bias=True)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.number_of_nodes()
        self.step(n)
        return self.x, self.y, self.parent

    def cost(self):
        return self.cost

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
    goal = (random.randint(0, dimensions[0]- 1), random.randint(50, dimensions[0] - 1))
    obsdim = 60
    obsnum = 50
    iteration = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    graph = RRTGraph(start, goal, dimensions, map.map)

    obstacles = map.makeobs()
    # confined = [    ('rect', (794, 5, 45, 39)),    ('triangle', [(918, 364), (978, 364), (941, 315)]),    ('triangle', [(230, 98), (272, 98), (254, 46)]),    ('rect', (323, 226, 40, 56)),    ('triangle', [(670, 547), (724, 547), (695, 505)]),    ('triangle', [(873, 359), (931, 359), (901, 307)]),    ('circle', ((617, 232), 28)),    ('rect', (645, 478, 54, 57)),    ('circle', ((168, 243), 21)),    ('rect', (438, 427, 55, 37)),    ('circle', ((590, 551), 27)),    ('rect', (588, 23, 43, 41)),    ('triangle', [(876, 465), (925, 465), (900, 412)]),    ('circle', ((324, 554), 17)),    ('rect', (386, 363, 51, 32)),    ('circle', ((418, 60), 17)),    ('rect', (105, 84, 52, 37)),    ('triangle', [(54, 84), (102, 84), (83, 53)]),    ('circle', ((541, 567), 28)),    ('circle', ((830, 291), 20)),    ('rect', (845, 88, 50, 39)),    ('triangle', [(916, 8), (956, 8), (943, -44)]),    ('circle', ((359, 406), 28)),    ('triangle', [(175, 416), (235, 416), (198, 369)]),    ('circle', ((412, 206), 26)),    ('rect', (380, 45, 49, 31)),    ('circle', ((440, 264), 22)),    ('rect', (174, 461, 45, 53)),    ('circle', ((310, 538), 24)),    ('rect', (68, 565, 45, 44)),    ('rect', (653, 196, 60, 40)),    ('triangle', [(376, 449), (422, 449), (391, 394)]),    ('triangle', [(0, 319), (48, 319), (16, 264)]),    ('rect', (565, 234, 42, 34)),    ('circle', ((530, 184), 26)),    ('triangle', [(728, 112), (766, 112), (745, 52)]),    ('circle', ((683, 339), 15)),    ('triangle', [(864, 140), (919, 140), (887, 93)]),    ('triangle', [(636, 163), (670, 163), (664, 118)]),    ('rect', (40, 2, 35, 54)),    ('triangle', [(931, 507), (971, 507), (958, 456)]),    ('triangle', [(339, 493), (387, 493), (366, 458)]),    ('rect', (135, 552, 45, 45)),    ('circle', ((245, 517), 16)),    ('circle', ((350, 181), 15)),    ('circle', ((119, 187), 30)),    ('circle', ((223, 151), 29)),    ('rect', (92, 108, 53, 39)),    ('rect', (327, 267, 31, 57)),    ('triangle', [(798, 174), (835, 174), (823, 133)])]
    # obstacles = confined
    map.drawMap(obstacles)
    graph.cache_obstacle_grid()

    pygame.display.update()
    pygame.event.clear()

    if waitClick(): print(obstacles)
    
    distToGoal = math.dist(start, goal)
    start_time = pygame.time.get_ticks()
    font = pygame.font.Font(None, 36)

    isSolved = False
    graph.bias()
    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - start_time) / 1000.0
        if graph.num_not_in_ellipse and graph.ellipse:
            area = graph.ellipse.get_area()
            if area and ((graph.number_of_nodes() - graph.num_not_in_ellipse) / max(math.log(area + 1), 0.8) > 600):
                break
        # if iteration % 1000 == 0:
        #     X, Y, Parent = graph.bias()
        #     pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad, map.nodeRad + 4, 0)
        #     pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        # else:
        X, Y, Parent = graph.expand()
        #     pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad+4)
        #     pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        if iteration % 500 == 0:
            graph.updateKDTree()

        
        if not isSolved: isSolved = graph.path_to_goal()
        if iteration % 1000 == 0:
            map.map.fill((255,255,255))
            map.drawMap(obstacles)
            
            for i in range(len(X)):
                pygame.draw.circle(map.map, map.grey if (X[i] != goal[0] or Y[i] != goal[1]) and (X[i] != start[0] or Y[i] != start[1]) else map.Red, (X[i], Y[i]), map.nodeRad+1)
                if Parent[i] is not None:
                    pygame.draw.line(map.map, map.Blue, (X[i], Y[i]), (X[Parent[i]], Y[Parent[i]]), map.edgeThickness)
            timer_text = font.render(f"Time: {elapsed_time:.2f} s", True, (0, 0, 0))
            map.map.blit(timer_text, (10, 10))
            if isSolved:
                graph.path_to_goal()
                pygame.display.update()
                path = graph.getPathCoords()
                graph.ellipse.draw_bounding_rectangle(map.map)
                map.drawPath(path)
                if abs(graph.costs[graph.best] - distToGoal) < 1:
                    pygame.display.update()
                    break

            pygame.display.update()
            # waitClick()
            graph.path = []

        iteration += 1
    
    current_time = pygame.time.get_ticks()
    elapsed_time = (current_time - start_time) / 1000.0
    timer_text = font.render(f"Total Time: {elapsed_time:.2f} s", True, (16, 150, 34))
    map.map.blit(timer_text, (10, 30))
    map.drawPath(graph.getPathCoords())
    pygame.display.update()
    pygame.event.clear()
    if waitClick(): print(obstacles)
    main()


if __name__ == "__main__":
    main()

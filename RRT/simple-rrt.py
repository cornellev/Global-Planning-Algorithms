import random
import math
import pygame
import time


class RRTMap:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.Maph, self.Mapw = self.MapDimensions

        self.MapWindowName = "RRT Path Planning"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.Mapw, self.Maph))
        self.map.fill((255, 255, 255))
        self.nodeRad = 0
        self.nodeThickness = 0
        self.edgeThickness = 1

        self.obstacles = []
        self.obsdim = obsdim
        self.obsnum = obsnum

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
        for node in path:
            pygame.draw.circle(self.map, self.Red, node, self.nodeRad+3, 0)

    def drawObs(self, obstacles):
        for obs in obstacles:
            shape, properties = obs
            if shape == "rect":
                pygame.draw.rect(self.map, self.black, properties)
            elif shape == "circle":
                pygame.draw.circle(self.map, self.black, properties[0], properties[1])
            elif shape == "triangle":
                pygame.draw.polygon(self.map, self.black, properties)


class RRTGraph:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum, surface):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.MapDimensions = MapDimensions
        self.maph, self.mapw = self.MapDimensions
        self.x = [x]
        self.y = [y]
        self.parent = [0]
        self.costs = [0]
        self.surface = surface

        self.obstacles = []
        self.obsDim = obsdim
        self.obsNum = obsnum

        self.goalstate = None
        self.path = []

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

        for _ in range(self.obsNum):
            shape = None
            startGoalCol = True
            while startGoalCol:
                shape = self.makeRandomShape()
                shape_type, properties = shape
                if shape_type == "rect":
                    startGoalCol = properties.collidepoint(self.goal) or properties.collidepoint(self.start)
                elif shape_type == "circle":
                    startGoalCol = pygame.Rect(properties[0][0] - properties[1], properties[0][1] - properties[1],
                                               2 * properties[1], 2 * properties[1]).collidepoint(self.goal) or \
                                   pygame.Rect(properties[0][0] - properties[1], properties[0][1] - properties[1],
                                               2 * properties[1], 2 * properties[1]).collidepoint(self.start)
                elif shape_type == "triangle":
                    triangle_rect = pygame.Rect(min(p[0] for p in properties), min(p[1] for p in properties),
                                                self.obsDim, self.obsDim)
                    startGoalCol = triangle_rect.collidepoint(self.goal) or triangle_rect.collidepoint(self.start)

            obs.append(shape)

        self.obstacles = obs.copy()
        return obs

    def add_node(self, n, x, y):
        self.x.insert(n,x)
        self.y.insert(n,y)

    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    def remove_edge(self, n):
        self.parent.pop(n)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self, n1, n2):
        return math.sqrt((float(self.x[n1]) - float(self.x[n2])) ** 2 + (float(self.y[n1]) - float(self.y[n2])) ** 2)

    def sample_envir(self):
        return random.randint(0, self.mapw - 1), random.randint(0, self.maph - 1)

    def nearest(self, node):
        dmin = self.distance(0, node)
        n, d = 0, 0
        for i in range(1, node):
            d = self.distance(i, node)
            if d < dmin:
                dmin = d
                n = i
        return n

    def isFree(self):
        n = self.number_of_nodes() - 1
        color_at_node = pygame.Surface.get_at(self.surface, (self.x[n], self.y[n]))
        return color_at_node != (0, 0, 0, 255)

    def cross_obstacle(self, startNode, endNode):
        start_pos = (self.x[startNode], self.y[startNode])
        end_pos = (self.x[endNode], self.y[endNode])
        
        line_length = math.dist(start_pos, end_pos)
        num_points = max(2, round(line_length / 3))
        
        points_on_line = self.get_line_points(start_pos, end_pos, num_points)
        
        for point in points_on_line:
            if pygame.Surface.get_at(self.surface, point) == (0, 0, 0, 255):
                return True
        return False

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

    def connect(self, nearest, node):
        if self.cross_obstacle(nearest, node):
            self.remove_node(node)
            return False
        self.add_edge(nearest, node)
        return True

    def step(self, nearest, node, dmax=35):
        dist = self.distance(nearest, node)
        foundGoal = False
        if dist > dmax:
            dx = self.x[node] - self.x[nearest]
            dy = self.y[node] - self.y[nearest]
            theta = math.atan2(dy, dx)
            x = int(self.x[nearest] + dmax * math.cos(theta))
            y = int(self.y[nearest] + dmax * math.sin(theta))
            self.remove_node(node)
            if abs(x - self.goal[0]) < dmax and abs(y - self.goal[1]) < dmax:
                self.add_node(node, self.goal[0], self.goal[1])
                foundGoal = True
            else:
                self.add_node(node, x, y)
        else:
            foundGoal = abs(self.x[node] - self.goal[0]) < dmax and abs(self.y[node] - self.goal[1]) < dmax
        if self.connect(nearest, node) and foundGoal:
            self.goalstate = node
            self.goalFlag = True

    def path_to_goal(self):
        if self.goalFlag:
            self.path = []
            self.path.append(self.goalstate)
            newpoint = self.parent[self.goalstate]
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

    def bias(self, goalNode):
        n = self.number_of_nodes()
        self.add_node(n, goalNode[0], goalNode[1])
        nearest = self.nearest(n)
        self.step(nearest, n)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.number_of_nodes()
        x,y = self.sample_envir()
        self.add_node(n, x, y)
        if self.isFree():
            nearest = self.nearest(n)
            self.step(nearest, n)
        else:
            self.remove_node(n)
        return self.x, self.y, self.parent

    def cost(self):
        return self.cost


def main():
    dimensions = (600, 1000)
    start = (50, 50)
    goal = (910, 510)
    obsdim = 60
    obsnum = 50
    iteration = 0

    pygame.init()
    map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    graph = RRTGraph(start, goal, dimensions, obsdim, obsnum, map.map)

    obstacles = graph.makeobs()

    map.drawMap(obstacles)
     
    pygame.display.update()
    pygame.event.clear()
    click = False
    while not click:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN: click = True
       

    while not graph.path_to_goal():
        if iteration % 10 == 0:
            X, Y, Parent = graph.bias(goal)
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad, map.nodeRad + 4, 0)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        else:
            X, Y, Parent = graph.expand()
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad+4)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        if iteration % 1 == 0:
            pygame.display.update()

        iteration += 1

    map.drawPath(graph.getPathCoords())
    pygame.display.update()
    pygame.event.clear()
    click = False
    while not click:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN: click = True
    main()


if __name__ == "__main__":
    main()

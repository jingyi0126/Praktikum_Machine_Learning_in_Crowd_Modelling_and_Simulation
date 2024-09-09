import scipy.spatial.distance
import numpy as np
import numpy.typing as npt

from src import elements as el, utils
import random
import heapq


# import pandas as pd

class Simulation:

    def __init__(self, config: el.SimulationConfig, random_seed: int = 42):
        """
    Initializes a new instance of the simulation environment with the specified configuration settings.

    This constructor sets up the simulation environment by initializing various attributes such as grid size,
    output filename, distance computation method, and other configurations provided in the SimulationConfig object.
    It also initializes state-related attributes that will keep track of pedestrians, targets, obstacles,
    measuring points, pedestrian speeds, and more, to manage the simulation state across steps.

    Parameters:
    - config (el.SimulationConfig): An object containing all configuration settings required to initialize the simulation,
      such as grid size, output file name, and lists of pedestrians, targets, obstacles, and measuring points.
    - random_seed (int, optional): A seed value for the random number generator to ensure reproducibility of the simulation. 
      Defaults to 42.

    Attributes:
    - width (int): Width of the simulation grid.
    - height (int): Height of the simulation grid.
    - output_filename (str): The name of the file where simulation outputs are stored.
    - distance_computation (str): The method used for computing distances in the simulation.
    - grid_size (Tuple[int, int]): The size of the grid as a tuple of width and height.
    - pedestrians (list): List of pedestrian objects participating in the simulation.
    - targets (list): List of target points that pedestrians aim to reach.
    - obstacles (list): List of obstacle objects in the grid.
    - measuring_points (list): List of measuring points for collecting simulation data.
    - is_absorbing (bool): Flag to indicate whether the grid has absorbing boundaries.
    - init_speeds (dict): Dictionary mapping pedestrian IDs to their initial speeds.
    - real_time_pedestrians_int (dict): Dictionary mapping pedestrian IDs to their current positions as integer coordinates.
    - real_time_pedestrians_float (dict): Dictionary mapping pedestrian IDs to their current positions as floating-point coordinates.
    - dist_grid (array): Grid representing the distance to the nearest target for each cell.
    - pedestrians_at_target (list): List of pedestrian IDs that have reached their target.
    - measuring_records (dict): Dictionary mapping measuring point IDs to lists of collected data points.
    - recorded_pedestrians (dict): Dictionary mapping measuring point IDs to sets of pedestrian IDs that have been recorded at each point.
    - average_flows (dict): Dictionary mapping measuring point IDs to the average flow rate of pedestrians through the point.
    - steps (int): Counter for the number of simulation steps taken.
    - flows_records (dict): Dictionary holding the records of pedestrian flows.
    - ID_sequence (list): Sequential list of pedestrian IDs as they reach their targets.
    - travel_time (list): List capturing the travel times of pedestrians.

    The constructor also computes the initial distance grid based on target locations and initializes various tracking
    lists and dictionaries for managing simulation state during the execution.
    """

        self.width, self.height = config.grid_size.width, config.grid_size.height
        self.output_filename = config.output_filename

        # TODO: initialize other fields.
        self.distance_computation = config.distance_computation
        self.grid_size = config.grid_size
        self.pedestrians = config.pedestrians
        self.targets = config.targets
        self.obstacles = config.obstacles
        self.measuring_points = config.measuring_points
        # print(self.measuring_points)
        self.is_absorbing = config.is_absorbing
        """Initialize the initial speeds of the pedestrians and the measuring time."""
        self.init_speeds = {}
        self.real_time_pedestrians_int = {}
        self.real_time_pedestrians_float = {}
        self.dist_grid = self._compute_distance_grid(self.targets)
        self.pedestrians_at_target = []
        self.measuring_records = {}
        self.recorded_pedestrians = {}
        self.average_flows = {}
        self.steps = 0
        self.flows_records = {}
        for pedestrian in self.pedestrians:
            pedestrian.took_steps = 0
            self.init_speeds[pedestrian.ID] = pedestrian.speed
            self.real_time_pedestrians_int[pedestrian.ID] = (pedestrian.x, pedestrian.y)
            self.real_time_pedestrians_float[pedestrian.ID] = (pedestrian.x, pedestrian.y)

        if self.measuring_points != []:
            for measuring_point in self.measuring_points:
                self.measuring_records[measuring_point.ID] = []
                self.recorded_pedestrians[measuring_point.ID] = set()
                self.average_flows[measuring_point.ID] = 0.0
        self.ID_sequence = []  # A list contains the IDs of the pedestrians, dependent of the order in which pedestrians arrive at the destination
        self.travel_time = []

    def update(self, perturb: bool = True) -> bool:
        """Performs one step of the simulation.

        Arguments:
        ----------
        perturb : bool
            If perturb=False, pedestrians' positions are updates in the
            fixed order. Otherwise, the pedestrians are shuffle before
            performing an update.

        Returns:
        --------
        bool
            True if all pedestrians reached a target and the simulation
            is over, False otherwise.
        """
        print(f"New Update{self.steps}")
        # for measuring_point in self.measuring_points:
        #     self.measuring_records[measuring_point.ID] = []
        #     self.recorded_pedestrians[measuring_point.ID] = set()
        # TODO: implement the update.
        if perturb:
            random.shuffle(self.pedestrians)

        finished = True

        """If we shuffle the neighbors."""
        shuffle = True

        """Pedestrian step somulation."""
        blocks = set()
        for obstacle in self.obstacles:
            blocks.add((obstacle.x, obstacle.y))

        if self.measuring_points != []:
            for measuring_point in self.measuring_points:
                self.recorded_pedestrians[measuring_point.ID] = set()

        for pedestrian in self.pedestrians:
            # pedestrian.took_steps += 1
            # print(f"ID:{pedestrian.ID}, steps: {pedestrian.took_steps}")
            if self.at_target((pedestrian.x, pedestrian.y), self.targets):
                self.pedestrians_at_target.append(pedestrian)
                # print(f"ID:{pedestrian.ID}, total_steps: {pedestrian.took_steps + 1}")
                # self.ID_sequence.append(pedestrian.ID)
                # self.travel_time.append(pedestrian.took_steps + 1)
                if self.is_absorbing:
                    self.pedestrians.remove(pedestrian)
                    self.real_time_pedestrians_float.pop(pedestrian.ID)
                    self.real_time_pedestrians_int.pop(pedestrian.ID)

            if pedestrian in self.pedestrians_at_target:
                continue

            origin_x = self.real_time_pedestrians_float[pedestrian.ID][0]
            origin_y = self.real_time_pedestrians_float[pedestrian.ID][1]

            self.near_target_adjustment(pedestrian)

            (min_neighbor, min_neighbor_avoidance_cost) = self.find_best_neighbor(pedestrian, blocks, shuffle)

            # self.speed_adjustment(pedestrian,min_neighbor_avoidance_cost)

            ##############################################################################################
            # Q5:
            # How should we implement the speed adjustment? Does the existing one meet the requirement?
            ##############################################################################################

            # If the pedestrians are not moving, sin and cos should be 0.0 since they're meaningless.
            if not (min_neighbor.x == pedestrian.x and min_neighbor.y == pedestrian.y):
                cos_theta = (min_neighbor.x - pedestrian.x) / np.sqrt(
                    (min_neighbor.x - pedestrian.x) ** 2 + (min_neighbor.y - pedestrian.y) ** 2)
                sin_theta = (min_neighbor.y - pedestrian.y) / np.sqrt(
                    (min_neighbor.x - pedestrian.x) ** 2 + (min_neighbor.y - pedestrian.y) ** 2)
            else:
                cos_theta = 0.0
                sin_theta = 0.0

            next_position_real_x = self.real_time_pedestrians_float[pedestrian.ID][0] + cos_theta * pedestrian.speed
            next_position_real_y = self.real_time_pedestrians_float[pedestrian.ID][1] + sin_theta * pedestrian.speed

            # Check if the pedestrian will hit an obstacle or others, if so, move the pedestrian to the farthest possible cell on path.
            all_blocks = {value for key, value in self.real_time_pedestrians_int.items() if key != pedestrian.ID}
            all_blocks.update(blocks)
            safe_position = (next_position_real_x, next_position_real_y)
            path = self.bresenham_path((pedestrian.x, pedestrian.y),
                                       (round(next_position_real_x), round(next_position_real_y)))
            for i, position in enumerate(path):
                if position in all_blocks:
                    if i > 0:
                        safe_position = path[i - 1]  # Safe position is the last position
                    break
            (next_position_real_x, next_position_real_y) = safe_position

            # Final Boundary Check
            next_position_real_x = max(0, min(self.width - 1, next_position_real_x))
            next_position_real_y = max(0, min(self.height - 1, next_position_real_y))

            # Flow measuring

            "Detect by position"

            # if self.measuring_points!=[]:
            #     #For every measuring points, check the conditions
            #     for measuring_point in self.measuring_points:
            #         #Check the time
            #         if measuring_point.delay <= pedestrian.took_steps < measuring_point.delay + measuring_point.measuring_time:

            #             #Check if the position of the pedestrian is in the area of measuring points

            #             # if measuring_point.upper_left.x <= round(next_position_real_x) < measuring_point.upper_left.x + measuring_point.size.width and \
            #             # measuring_point.upper_left.y <= round(next_position_real_y) < measuring_point.upper_left.y + measuring_point.size.height:

            #             if measuring_point.upper_left.x <= pedestrian.x < measuring_point.upper_left.x + measuring_point.size.width and \
            #             measuring_point.upper_left.y <= pedestrian.y < measuring_point.upper_left.y + measuring_point.size.height:
            #                 #Check if the pedestrian is already recorded
            #                 #if pedestrian.ID not in self.recorded_pedestrians.values():
            #                     #Check if the pedestrian is moving
            #                 if pedestrian.x == round(next_position_real_x) and pedestrian.y == round(next_position_real_y):
            #                     self.measuring_records[measuring_point.ID].append((pedestrian.ID, (pedestrian.x, pedestrian.y), pedestrian.took_steps, 0.0))
            #                 else:
            #                     #Compute how far does the pedestrian really go
            #                     self.measuring_records[measuring_point.ID].append((pedestrian.ID, (pedestrian.x, pedestrian.y), pedestrian.took_steps, np.sqrt((next_position_real_x-origin_x)**2+(next_position_real_y-origin_y)**2)))
            #                 self.recorded_pedestrians[measuring_point.ID].add(pedestrian.ID)

            "Detect by path"

            if self.measuring_points != []:
                path = self.bresenham_path((pedestrian.x, pedestrian.y),
                                           (round(next_position_real_x), round(next_position_real_y)))
                for position in path:
                    # For every measuring points, check the conditions
                    for measuring_point in self.measuring_points:
                        # Check the time
                        if measuring_point.delay <= pedestrian.took_steps < measuring_point.delay + measuring_point.measuring_time:
                            # Check if the position of the pedestrian is in the area of measuring points. so even if the pedestrian is just going through the area ,it will be detected.

                            if measuring_point.upper_left.x <= position[
                                0] < measuring_point.upper_left.x + measuring_point.size.width and \
                                    measuring_point.upper_left.y <= position[
                                1] < measuring_point.upper_left.y + measuring_point.size.height:

                                # Check if the pedestrian is already recorded
                                if pedestrian.ID not in self.recorded_pedestrians.values():
                                    # Check if the pedestrian is moving
                                    if pedestrian.x == round(next_position_real_x) and pedestrian.y == round(
                                            next_position_real_y):
                                        self.measuring_records[measuring_point.ID].append(
                                            (pedestrian.ID, (pedestrian.x, pedestrian.y), pedestrian.took_steps, 0.0))
                                    else:
                                        # Compute how far does the pedestrian really go
                                        self.measuring_records[measuring_point.ID].append((pedestrian.ID,
                                                                                           (pedestrian.x, pedestrian.y),
                                                                                           pedestrian.took_steps,
                                                                                           np.sqrt((
                                                                                                           next_position_real_x - origin_x) ** 2 + (
                                                                                                           next_position_real_y - origin_y) ** 2)))
                                    self.recorded_pedestrians[measuring_point.ID].add(pedestrian.ID)

            # Update the position
            self.real_time_pedestrians_float[pedestrian.ID] = (next_position_real_x, next_position_real_y)
            pedestrian.x = round(next_position_real_x)
            pedestrian.y = round(next_position_real_y)
            self.real_time_pedestrians_int[pedestrian.ID] = (pedestrian.x, pedestrian.y)
            pedestrian.took_steps += 1

        if self.measuring_points != []:
            for measuring_point in self.measuring_points:
                if measuring_point.delay <= self.steps < measuring_point.delay + measuring_point.measuring_time:
                    self.flows_records[self.steps] = self.get_measured_flows()

                    ###########################################
                    # Q2:
                    # Should we compute the flow in every step?
                    ###########################################
        self.steps += 1

        # Uncomment the following codes for test 4 in task 5, which causes test error in Artemis.
        # for pedestrian in self.pedestrians:
        #     if self.at_target((pedestrian.x, pedestrian.y), self.targets):
        #         print(f"ID:{pedestrian.ID}, total_steps: {pedestrian.took_steps + 1}")
        #         self.ID_sequence.append(pedestrian.ID)
        #         self.travel_time.append(pedestrian.took_steps + 1)

        for pedestrian in self.pedestrians:
            # if pedestrian in self.pedestrians_at_target:
            #     continue
            """If the simulation is finished."""
            if not self.at_target((pedestrian.x, pedestrian.y), self.targets):
                finished = False
                break

        # if self.steps == 1:
        #     self._post_process()

        if finished:
            print("finished")
            self._post_process()
            return True
        return finished

    def find_best_neighbor(self, pedestrian: el.Pedestrian, blocks: set[tuple[int, int]], shuffle: bool) -> tuple[
        utils.Position, np.float64]:
        """
        Calculates the optimal neighboring position for a pedestrian by evaluating movement costs toward the target,
        avoiding obstacles, and considering pedestrian interactions.

        Parameters:
        - pedestrian (el.Pedestrian): The pedestrian whose next move is being determined.
        - blocks (set[tuple[int, int]]): Positions on the grid where movement is blocked.
        - shuffle (bool): If True, the neighbors are considered in random order to avoid deterministic patterns.

        Returns:
        - tuple[utils.Position, np.float64]: The best neighboring position and its associated cost. Returns the current
        position and infinite cost if no valid moves are found.

        
        """
        neighbors = self._get_neighbors(pedestrian, shuffle)
        neighbors_cost = {}
        neighbors_avoidance_cost = {}  # For speed adjustment
        sin_theta = 0.0
        cos_theta = 1.0
        existing_pedestrians = [p for p in self.pedestrians if p not in self.pedestrians_at_target]
        # Compute the cost of each neighbors
        for neighbor in neighbors:
            neighbors_cost[(neighbor.x, neighbor.y)] = np.inf  # Default value for impossible directions
            neighbors_avoidance_cost[(neighbor.x, neighbor.y)] = np.inf
        for neighbor in neighbors:
            cos_theta = (neighbor.x - pedestrian.x) / np.sqrt(
                (neighbor.x - pedestrian.x) ** 2 + (neighbor.y - pedestrian.y) ** 2)
            sin_theta = (neighbor.y - pedestrian.y) / np.sqrt(
                (neighbor.x - pedestrian.x) ** 2 + (neighbor.y - pedestrian.y) ** 2)
            if pedestrian.speed * cos_theta >= 1 or pedestrian.speed * sin_theta >= 1:
                next_possible_position = (round(pedestrian.x + cos_theta * pedestrian.speed),
                                          round(pedestrian.y + sin_theta * pedestrian.speed))

            else:
                next_possible_position = (
                    round(pedestrian.x) + int(np.sign(cos_theta)), round(pedestrian.y) + int(np.sign(sin_theta)))

            # Avoid obstacles
            blocked = False
            for position in self.bresenham_path((pedestrian.x, pedestrian.y), next_possible_position):
                if position in blocks:
                    neighbors_cost[(neighbor.x, neighbor.y)] = np.inf
                    blocked = True
                    break
            if blocked:
                continue

            # Avoid stepping back
            if 0 <= next_possible_position[0] < self.width and 0 <= next_possible_position[1] < self.height:
                gain_in_step = self.dist_grid[pedestrian.x, pedestrian.y] - self.dist_grid[
                    next_possible_position[0], next_possible_position[1]]
                if gain_in_step < 0:
                    continue  # a minus gain in step is walking away from the target
                neighbors_avoidance_cost[(neighbor.x, neighbor.y)] = self.avoidance_cost(pedestrian, utils.Position(
                    next_possible_position[0], next_possible_position[1]), existing_pedestrians)
                neighbors_cost[(neighbor.x, neighbor.y)] = neighbors_avoidance_cost[
                                                               (neighbor.x, neighbor.y)] - gain_in_step
        min_cost = np.inf
        min_neighbor = pedestrian
        best_found = False
        for neighbor in neighbors:
            if neighbors_cost[(neighbor.x, neighbor.y)] < min_cost:
                min_cost = neighbors_cost[(neighbor.x, neighbor.y)]
                min_neighbor = neighbor
                best_found = True
        # print(f"ID = {pedestrian.ID}, min_neighbor= ({min_neighbor.x},{min_neighbor.y})")
        # print(pedestrian.x,pedestrian.y)
        if best_found:
            return (min_neighbor, neighbors_avoidance_cost[(min_neighbor.x, min_neighbor.y)])
        else:
            return (pedestrian, np.inf)

    def near_target_adjustment(self, pedestrian: el.Pedestrian):
        """
        Adjusts the speed of a pedestrian if they are close to their target to prevent overshooting.
        This method modifies the pedestrian's speed to match the remaining distance to the target if that
        distance is less than their current speed.

        Parameters:
        - pedestrian (el.Pedestrian): The pedestrian whose speed may be adjusted.
        """
        distance = self.dist_grid[pedestrian.x, pedestrian.y]
        if distance < pedestrian.speed:
            pedestrian.speed = distance

    def speed_adjustment(self, pedestrian: el.Pedestrian, avoidance_cost: np.float64):
        """
        Modifies the speed of a pedestrian based on the surrounding environment's avoidance cost.
        This method increases the pedestrian's speed by a factor if there is no avoidance cost and the current
        speed is less than or equal to 120% of the initial speed. If there's an avoidance cost, the pedestrian's
        speed is reset to the initial speed.

        Parameters:
        - pedestrian (el.Pedestrian): The pedestrian whose speed is being adjusted.
        - avoidance_cost (np.float64): The cost associated with avoiding obstacles or other pedestrians, influencing speed adjustment.
        """
        if avoidance_cost == 0 and pedestrian.speed <= self.init_speeds[pedestrian.ID] * 1.2:
            pedestrian.speed *= 1.03  # log 1.03 (1.2) = 6.2

        if avoidance_cost != 0:
            pedestrian.speed = self.init_speeds[pedestrian.ID]

    def at_target(self, position: tuple[int, int], targets: tuple[utils.Position]) -> bool:
        """Returns a bool which indicates if a pedestrian is at the target."""
        at = False
        for target in targets:
            if position[0] == target.x and position[1] == target.y:
                at = True

        return at

    def avoidance_cost(self, pedestrian: el.Pedestrian, next_position: utils.Position,
                       pedestrians: list[el.Pedestrian]) -> np.float64:
        """
        Calculates and returns the avoidance cost for a pedestrian moving to a specified position,
        based on the proximity of other pedestrians along and adjacent to the path.

        Parameters:
        - pedestrian (el.Pedestrian): The pedestrian whose move is being evaluated.
        - next_position (utils.Position): The proposed next position for the pedestrian.
        - pedestrians (list[el.Pedestrian]): List of other pedestrians to consider for potential collisions.

        Returns:
        - np.float64: The total avoidance cost, determined by the closeness of other pedestrians to the direct and
        adjacent paths calculated using Bresenham's algorithm. The cost is exponentially higher as pedestrians are
        closer to the path.
        """

        next_direction = (int(np.sign(next_position.x - pedestrian.x)), int(np.sign(next_position.y - pedestrian.y)))
        next_direction_left45 = (
            int(np.sign(next_direction[0] - next_direction[1])), int(np.sign(next_direction[0] + next_direction[1])))
        next_direction_right45 = (
            int(np.sign(next_direction[0] + next_direction[1])), int(np.sign(next_direction[1] - next_direction[0])))
        r_max = 1.4
        total_avoidance_cost = 0.0
        path = self.bresenham_path((pedestrian.x, pedestrian.y), (next_position.x, next_position.y))
        path.extend(
            self.bresenham_path((pedestrian.x + next_direction_left45[0], pedestrian.y + next_direction_left45[1]),
                                (next_position.x, next_position.y)))
        path.extend(
            self.bresenham_path((pedestrian.x + next_direction_right45[0], pedestrian.y + next_direction_right45[1]),
                                (next_position.x, next_position.y)))

        for people in [p for p in pedestrians if p != pedestrian]:
            if (people.x, people.y) not in path:
                continue
            r = np.sqrt((people.x - next_position.x) ** 2 + (people.y - next_position.y) ** 2)

            endure_factor = 0.9  # If too many people is near the target, then they should endure each other to enter the target.

            if r <= r_max:
                added_cost = endure_factor * np.exp(
                    1 / ((r ** 2 - r_max ** 2) + 0.01))  # add 0.01 for the extreme case.
                total_avoidance_cost += added_cost

        return total_avoidance_cost

    def bresenham_path(self, origin: tuple[int, int], next_position: tuple[int, int]):
        """
        Generates and returns a list of grid points forming a straight line between two points using Bresenham's line algorithm.
        
        This method is commonly used in graphics to draw lines with a clear, well-defined pattern on a pixel grid,
        and is adapted here to determine the points along a direct path between two positions in a grid-based simulation.

        Parameters:
        - origin (tuple[int, int]): The starting point of the line.
        - next_position (tuple[int, int]): The ending point of the line.

        Returns:
        - list[tuple[int, int]]: A list of coordinates representing the points along the line from the origin to the next position.
        """
        path = []
        dx = abs(next_position[0] - origin[0])
        dy = abs(next_position[1] - origin[1])
        x, y = origin[0], origin[1]
        sx = -1 if origin[0] > next_position[0] else 1
        sy = -1 if origin[1] > next_position[1] else 1
        if dx > dy:
            err = dx / 2.0
            while x != next_position[0]:
                path.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != next_position[1]:
                path.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        path.append((x, y))
        return path

    def get_grid(self) -> npt.NDArray[el.ScenarioElement]:
        """Returns a full state grid of the shape (width, height)."""

        # TODO: return a grid for visualization.

        grid = np.full((self.width, self.height), el.ScenarioElement.empty)

        # for obstacle in self.obstacles:
        #     grid[obstacle.x,obstacle.y] = el.ScenarioElement.obstacle

        for pedestrian in self.pedestrians:
            """Pedestrians should be spawned in an empty grid in the window."""
            if 0 <= pedestrian.x < self.width and 0 <= pedestrian.y < self.height:
                if grid[pedestrian.x, pedestrian.y] == el.ScenarioElement.empty:
                    grid[pedestrian.x, pedestrian.y] = el.ScenarioElement.pedestrian

        for target in self.targets:
            if grid[target.x, target.y] != el.ScenarioElement.pedestrian:
                grid[target.x, target.y] = el.ScenarioElement.target

        for obstacle in self.obstacles:
            if grid[obstacle.x, obstacle.y] != el.ScenarioElement.pedestrian:
                grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle

        return grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""

        # TODO: return a distance grid.
        distance_grid = self._compute_distance_grid(self.targets)
        return distance_grid

    def get_measured_flows(self) -> dict[int, float]:
        """Returns a map of measuring points' ids to their flows.

        Returns:
        --------
        dict[int, float]
            A dict in the form {measuring_point_id: flow}.
        """
        measured_flow = {}
        # print(self.measuring_records)
        for measuring_point in self.measuring_points:
            if not (measuring_point.delay <= self.steps < measuring_point.delay + measuring_point.measuring_time):
                continue

            record_count = 0
            total_speed = 0.0
            # print(self.measuring_records[measuring_point.ID])

            for record in self.measuring_records[measuring_point.ID]:
                if record[2] == self.steps:
                    record_count += 1
                    total_speed += record[3]

            if record_count > 0:
                average_speed = total_speed / record_count
            else:
                average_speed = 0.0

            flow = average_speed * 0.4 * record_count / (
                    measuring_point.size.width * 0.4 * measuring_point.size.height * 0.4)

            # Compute the average flow
            measured_flow[measuring_point.ID] = (flow + self.average_flows[measuring_point.ID] * (
                    self.steps - measuring_point.delay)) / (self.steps - measuring_point.delay + 1)

            self.average_flows[measuring_point.ID] = measured_flow[measuring_point.ID]
            #####################################################################
            # Q3:
            # Is this the correct way for computing the flow? Should we compute the average flow over the whole
            # measuing time? For example, (flow(t1) + ... +flow(tn))/n, where tn - t1 = mearsuring time
            #####################################################################

            #####################################################################
            # Q4:
            # What kind of unit should we actually use?
            #####################################################################

        # print(f"Measured Flow = {measured_flow}")
        # print(measured_flow)
        return measured_flow

    def _compute_distance_grid(
            self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:

        """
        Computes and returns a grid of distances from each point on the grid to the nearest target, using
        the specified distance computation algorithm.

        Parameters:
        - targets (tuple[utils.Position]): A tuple of positions representing the targets on the grid. If no targets are
        specified, the method returns a grid of zeros.

        Returns:
        - npt.NDArray[np.float64]: A 2D numpy array where each element represents the distance from that grid point to the
        nearest target based on the selected algorithm.
        """

        if len(targets) == 0:
            distances = np.zeros((self.width, self.height))
            return distances

        match self.distance_computation:
            case "naive":
                distances = self._compute_naive_distance_grid(targets)
            case "dijkstra":
                distances = self._compute_dijkstra_distance_grid(targets)
            case _:
                print(
                    "Unknown algorithm for computing the distance grid: "
                    f"{self.distance_computation}. Defaulting to the "
                    "'naive' option."
                )
                distances = self._compute_naive_distance_grid(targets)
        return distances

    def _compute_naive_distance_grid(
            self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid without considering obstacles.

        Arguments:
        ----------
        targets : Tuple[utils.Position]
            A tuple of targets on the grid. For each cell, the algorithm
            computes the distance to the closes target.

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of distances of the same shape as the main grid.
        """

        targets = [[*target] for target in targets]
        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)
        distances = distances.reshape((self.height, self.width)).T

        return distances

    def _compute_dijkstra_distance_grid(
            self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """
        Computes and returns a grid of minimum distances to the nearest target using Dijkstra's algorithm.

        This method initializes a distance grid with infinite values, then applies Dijkstra's algorithm to calculate
        the shortest paths from each target to all other points on the grid. The distance calculation considers both
        orthogonal and diagonal moves, with diagonal moves having a higher cost.

        Parameters:
        - targets (tuple[utils.Position]): A tuple of target positions from which distances are calculated.

        Returns:
        - npt.NDArray[np.float64]: A 2D numpy array with each element representing the minimum distance from that grid
        point to the nearest target.
        
        """

        # TODO: implement the Dijkstra algorithm.

        distances = np.full((self.width, self.height), np.inf)
        edge = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        obstacles = set()

        for obstacle in self.obstacles:
            obstacles.add((obstacle.x, obstacle.y))

        for target in targets:
            distances[target.x, target.y] = 0
            heapq.heappush(edge, (0, (target.x, target.y)))

        while edge:
            min_node_value, min_node = heapq.heappop(edge)
            min_node_x, min_node_y = min_node

            for dx, dy in directions:
                new_x, new_y = min_node_x + dx, min_node_y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height and (new_x, new_y) not in obstacles:
                    cost = 1.414213562373095 if abs(dx) + abs(dy) == 2 else 1
                    if distances[new_x, new_y] > min_node_value + cost and (new_x, new_y) not in obstacles:
                        distances[new_x, new_y] = min_node_value + cost
                        heapq.heappush(edge, (distances[new_x, new_y], (new_x, new_y)))
        return distances

    def _get_neighbors(
            self, position: utils.Position, shuffle: bool = True
    ) -> list[utils.Position]:
        """Returns a list of neighboring cells for the position.

        Arguments:
        ----------
        positions : utils.Position
            A position on the grid.
        shuffle : bool
            An indicator if neighbors should be shuffled or returned
            in the fixed order.

        Returns:
        --------
        list[utils.Position]
            An array of neighboring cells. Two cells are neighbors
            if they share a common vertex.
        """

        # TODO: return all neighbors.
        neighbors = []
        x_bias = [-1, 0, 1, -1, 1, -1, 0, 1]
        y_bias = [1, 1, 1, 0, 0, -1, -1, -1]
        """Boundary Check"""
        for i in range(8):
            if 0 <= position.x + x_bias[i] < self.width and 0 <= position.y + y_bias[i] < self.height:
                neighbors.append(utils.Position(position.x + x_bias[i], position.y + y_bias[i]))
        if shuffle:
            random.shuffle(neighbors)
        return neighbors

    def _post_process(self):
        """
        Processes the recorded flow records to calculate and print average values for each key.

        """

        print(self.flows_records)
        # Initialize a dictionary to hold sum of values and counts for each key
        totals = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
        # Sum the values for each key and count the occurrences
        for key, sub_dict in self.flows_records.items():
            for sub_key, value in sub_dict.items():
                totals[sub_key][0] += value
                totals[sub_key][1] += 1

        # Calculate the averages and round to 4 decimal places
        averages = {k: round(v[0] / v[1], 4) if v[1] != 0 else 0 for k, v in totals.items()}
        print(averages)
        # with open('outputs/flows_records.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Step', 'Measuring Point ID', 'Flow'])

        #     for step, records in self.flows_records.items():
        #         for point_id, flow in records.items():
        #             writer.writerow([step, point_id, flow])
        # TODO: store output for analysis.
        # if self.steps <= 40:
        #     ID = []
        #     speed = []
        # 
        #     for pedestrian in self.pedestrians:
        #         # print(pedestrian)
        #         ID.append(pedestrian.ID)
        #         speed.append(pedestrian.speed)
        #     print(speed)
        # df = pd.DataFrame({'ID': ID, 'speed': speed})
        # df.sort_values(by=['ID'], inplace=True)
        # df.reset_index(drop=True, inplace=True)
        # df.to_csv('outputs/age_speed.csv')
        '''Uncomment the following codes for test 4 in task 5, which causes test error in Artemis.'''
        # Sort the lists by ID
        # sorted_ID = sorted(self.ID_sequence)
        # sorted_travel_time = [self.travel_time[self.ID_sequence.index(id)] for id in sorted_ID]
        #
        # with open('outputs/travel_time_test4.csv', 'w') as f:
        #     f.write("ID,travel_time\n")
        #     for i in range(len(sorted_ID)):
        #         f.write(f"{sorted_ID[i]},{sorted_travel_time[i]}\n")

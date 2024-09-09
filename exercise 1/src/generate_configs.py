import json
import numpy as np
import os
import pandas as pd

CONFIG_FOLDER = "configs"
OUTPUT_FOLDER = "outputs"

CELL_SIZE = 0.4


def get_vertical_object(x: int, y_start: int, y_end: int) -> tuple[dict]:
    """Generates a list of positions in the provided vertical range.

    Parameters:
    -----------
    x : int
        The x coordinate of the vertical object.
    y_start : int
        The y coordinate of the lower end of the object.
    y_end : int
        The y coordinate of the upped end of the object.

    Returns:
    --------
    tuple[dict]
        A list of dictionaries with keys ('x', 'y') specifying positions
        of the object entries. Both ends at y_start and y_end are
        included to the object.
    """
    return tuple({"x": x, "y": y} for y in range(y_start, y_end + 1))


def get_horizontal_object(x_start: int, x_end: int, y: int) -> tuple[dict]:
    """Generates a list of positions in the provided horizontal range.

    Parameters:
    -----------
    x_start : int
        The x coordinate of the left end of the object.
    x_end : int
        The x coordinate of the right end of the object.
    y : int
        The y coordinate of the horizontal object.

    Returns:
    --------
    tuple[dict]
        A list of dictionaries with keys ('x', 'y') specifying positions
        of the object entries. Both ends at x_start and x_end are
        included to the object.
    """
    return tuple({"x": x, "y": y} for x in range(x_start, x_end + 1))


def generate_pedestrians(
        hor_span: tuple[int, int],
        vert_span: tuple[int, int],
        n_pedestrians: int,
        speed_bounds: tuple[float, float] = (1, 1),
        random_seed: int = 42,
) -> list[dict]:
    """Generates pedestrians in the specified rectangular area.

    Parameters:
    -----------
    hor_span : Tuple[int, int]
        A tuple (x_min, x_max) specifying horizontal borders of the
        spawn area.
    vert_span : Tuple[int, int]
        A tuple (y_min, y_max) specifying vertical borders of the
        spawn area.
    n_pedestrians : int
        The number of pedestrians to generate. The positions of the
        pedestrians are sampled from the uniform distribution.
    speed_bounds : Tuple[float, float] = (1, 1)
        A tuple (speed_min, speed_max) specifying speed bounds of
        pedestrians. The unit of speed is cells/step. A speed value for
        each pedestrian is sampled from the uniform distribution.
    random_seed : int = 42
        The random seed used to define a random generator.

    Returns:
    --------
    list[dict]
        A list of dictionaries with keys ('ID', 'x', 'y', 'speed')
        specifying the initial configuration of pedestrians in the
        simulation.
    """
    rng = np.random.default_rng(random_seed)
    spawn_width = hor_span[1] - hor_span[0] + 1
    spawn_height = vert_span[1] - vert_span[0] + 1

    positions = rng.choice(
        spawn_width * spawn_height, size=n_pedestrians, replace=False
    )
    xs = positions % spawn_width + hor_span[0]
    ys = positions // spawn_width + vert_span[0]
    speeds = rng.uniform(*speed_bounds, size=n_pedestrians)
    pedestrians = []
    for i, (x, y, speed) in enumerate(zip(xs, ys, speeds)):
        pedestrians.append({"ID": i, "x": int(x), "y": int(y), "speed": speed})
    return pedestrians


def save_json(
        filename: str,
        grid_size: dict,
        targets: tuple[dict],
        measuring_points: tuple[dict],
        obstacles: tuple[dict],
        pedestrians: tuple[dict],
        is_absorbing: bool,
        distance_computation: str,
        output_filename: str,
):
    """Saves the simulation configuration to a .json file."""

    config = {
        "grid_size": grid_size,
        "targets": targets,
        "obstacles": obstacles,
        "pedestrians": pedestrians,
        "measuring_points": measuring_points,
        "is_absorbing": is_absorbing,
        "distance_computation": distance_computation,
        "output_filename": output_filename,
    }
    with open(filename, "w") as fout:
        json.dump(config, fout, sort_keys=True, indent=4)


def task_1(filename: str):
    """Saves a configuration file for the Task 1."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 5, "height": 5}
    targets = [{"x": 3, "y": 2}]
    measuring_points = []
    obstacles = []
    pedestrians = [{"ID": 1, "x": 1, "y": 1, "speed": 1}]
    is_absorbing = False
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


# TODO: create configs for other tasks.

def task_2(filename: str):
    """Saves a configuration file for the Task 2."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 50, "height": 50}
    targets = [{"x": 25, "y": 25}]
    measuring_points = []
    obstacles = []
    pedestrians = [{"ID": 1, "x": 5, "y": 25, "speed": 1}]
    is_absorbing = False
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def task_3(filename: str):
    """Saves a configuration file for the Task 3."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 100, "height": 100}
    targets = [{"x": 50, "y": 50}]
    measuring_points = []
    obstacles = []
    pedestrians = [{"ID": 1, "x": 50, "y": 0, "speed": 5},
                   {"ID": 2, "x": 98, "y": 35, "speed": 5},
                   {"ID": 3, "x": 79, "y": 91, "speed": 5},
                   {"ID": 4, "x": 21, "y": 91, "speed": 5},
                   {"ID": 5, "x": 2, "y": 35, "speed": 5}]
    is_absorbing = True
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def chicken_test(filename: str):
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 70, "height": 50}
    targets = [{"x": 65, "y": 25}]
    measuring_points = []
    obstacles = get_vertical_object(x=50, y_start=11, y_end=40) + get_horizontal_object(x_start=35, x_end=50, y=10) + \
                get_horizontal_object(x_start=35, x_end=49, y=40)
    pedestrians = generate_pedestrians(hor_span=(5, 20), vert_span=(17, 33), n_pedestrians=150)
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def bottleneck(filename: str):
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 75, "height": 35}
    targets = get_vertical_object(x=68, y_start=17, y_end=19)
    # [{"x": 68, "y": 18}, {"x": 68, "y": 17}, {"x": 68, "y": 19}]
    measuring_points = []
    obstacles = get_vertical_object(x=5, y_start=5, y_end=30) + get_horizontal_object(x_start=6, x_end=30, y=5) + \
                get_horizontal_object(x_start=6, x_end=30, y=30) + get_vertical_object(x=30, y_start=6, y_end=16) + \
                get_vertical_object(x=30, y_start=20, y_end=29) + get_horizontal_object(x_start=31, x_end=43, y=16) + \
                get_horizontal_object(x_start=31, x_end=43, y=20) + get_vertical_object(x=43, y_start=5, y_end=15) + \
                get_vertical_object(x=43, y_start=21, y_end=30) + get_horizontal_object(x_start=44, x_end=68, y=5) + \
                get_horizontal_object(x_start=44, x_end=68, y=30) + get_vertical_object(x=68, y_start=6, y_end=16) + \
                get_vertical_object(x=68, y_start=20, y_end=29)
    pedestrians = generate_pedestrians(hor_span=(6, 17), vert_span=(6, 29), n_pedestrians=150)
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def test1(filename: str):
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 110, "height": 15}
    targets = get_vertical_object(x=104, y_start=6, y_end=10)
    # [{"x": 104, "y": 8}, {"x": 104, "y": 6}, {"x": 104, "y": 7}, {"x": 104, "y": 9}, {"x": 104, "y": 10}]  #
    measuring_points = []
    obstacles = get_horizontal_object(x_start=5, x_end=104, y=5) + get_horizontal_object(x_start=5, x_end=104, y=11)
    pedestrians = [{"ID": 1, "x": 5, "y": 8, "speed": 3.33}]  # 1.33(m/s) / CELL_SIZE(m)
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def test2(filename: str, density):
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 41, "height": 22}
    targets = get_vertical_object(x=40, y_start=2,
                                  y_end=20)  # {"x": 104, "y": 6}, {"x": 104, "y": 7},, {"x": 104, "y": 9}, {"x": 104, "y": 10}
    measuring_points = [
        {"ID": 1, "upper_left": {'x': 29, 'y': 10}, "size": {'width': 3, 'height': 3}, "delay": 2, "measuring_time": 6},
        {"ID": 2, "upper_left": {'x': 35, 'y': 10}, "size": {'width': 3, 'height': 3}, "delay": 2, "measuring_time": 6},
        {"ID": 3, "upper_left": {'x': 35, 'y': 13}, "size": {'width': 3, 'height': 3}, "delay": 2, "measuring_time": 6}]
    obstacles = get_horizontal_object(x_start=1, x_end=40, y=1) + get_horizontal_object(x_start=1, x_end=40, y=21)
    pedestrians = generate_pedestrians(hor_span=(1, 39), vert_span=(2, 20), n_pedestrians=int(38 * 18 * 0.16 * density),
                                       speed_bounds=(3, 3.5))
    # CELL_SIZE * CELL_SIZE =0.16; 3 ~ 3.5 cells/s represents 1.2 ~ 1.4 m/s
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def test3_pedestrians():
    """
    Generate a list of pedestrian dictionaries to meet the requirement of uniform distribution in test 3,
    each representing a pedestrian's position, ID, and speed.

    This function creates pedestrians on a grid. Pedestrians are positioned along y-values ranging from 31 to 35.
    For odd y-values, pedestrians are placed at x-values starting from 6 up to 18, spaced by 4 units.
    For even y-values, pedestrians are placed at x-values starting from 8 up to 20, also spaced by 4 units.
    Each pedestrian moves at a constant speed of 1.

    Returns:
        list of dicts: A list where each dictionary contains the ID, x-coordinate, y-coordinate,
                       and speed of a pedestrian. IDs are unique sequential integers starting from 1.
    """
    pedestrians = []
    k = 1
    for j in range(31, 36, 1):
        if j % 2 == 1:
            for i in range(6, 19, 4):
                pedestrian = {"ID": k, "x": i, "y": j, "speed": 1}
                pedestrians.append(pedestrian)
                k += 1
        elif j % 2 == 0:
            for i in range(8, 21, 4):
                pedestrian = {"ID": k, "x": i, "y": j, "speed": 1}
                pedestrians.append(pedestrian)
                k += 1
    return pedestrians


def test3(filename: str):
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 40, "height": 40}
    targets = get_horizontal_object(x_start=31, x_end=35, y=6)  # [{"x": 33, "y": 6}]
    measuring_points = []
    obstacles = get_vertical_object(x=5, y_start=30, y_end=36) + get_horizontal_object(x_start=6, x_end=30, y=30) + \
                get_vertical_object(x=30, y_start=6, y_end=29) + get_horizontal_object(x_start=6, x_end=36, y=36) + \
                get_vertical_object(x=36, y_start=6, y_end=35)
    pedestrians = test3_pedestrians()
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def sample():
    df = pd.read_csv('configs/rimea_7_speeds.csv')
    df = df.sample(n=50)
    df.reset_index(drop=True, inplace=True)
    df['ID'] = df.index + 1
    df['speed'] = df['speed']
    df.to_csv('outputs/samples.csv')
    pedestrians = []
    for i in range(1, 51):
        pedestrian = {"ID": i, "x": 2, "y": 1 + i, "speed": df.iloc[i - 1, 1]}
        pedestrians.append(pedestrian)
    return pedestrians


def test4(filename: str):
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 55, "height": 55}
    targets = get_vertical_object(x=52, y_start=2, y_end=51)
    measuring_points = []
    obstacles = []
    pedestrians = sample()
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


if __name__ == "__main__":
    task_1("toy_example")
    task_2("task_2")
    task_3("task_3")

    chicken_test("chicken_test")  # task_4
    bottleneck("bottleneck")  # task_4

    test1("test1")

    test2("test2_0.5", 0.5)
    test2("test2_1", 1)
    test2("test2_2", 2)
    test2("test2_3", 3)
    test2("test2_4", 4)
    test2("test2_5", 5)
    test2("test2_6", 6)

    test3("test3")
    test4("test4")

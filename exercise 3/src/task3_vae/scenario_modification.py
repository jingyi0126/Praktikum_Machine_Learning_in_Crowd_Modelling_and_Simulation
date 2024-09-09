import random


def get_targets(scenario: dict) -> list:
    """Returns a list of all targets in a scenario.

    Parameters:
    -----------
    scenario : dict
        The parsed scenario.

    Returns:
    --------
    list
        A list of the dictionaries of all targets.
    """

    return scenario["scenario"]["topography"]["targets"]


def get_attributes(scenario: dict) -> dict:
    """Returns the attributes of pedestrians.

    Parameters:
    -----------
    scenario : dict
        The parsed scenario.

    Returns:
    --------
    dict
        A dictionary of the attributes of pedestrians.
    """

    return scenario["scenario"]["topography"]["attributesPedestrian"]


def get_seed(scenario: dict) -> int | None:
    """Returns the seed used for the simulation of attributes.

    Parameters:
    -----------
    scenario : dict
        The parsed scenario.

    Returns:
    --------
    int | None
        The seed used for the simulation of attributes if one is used.
        Otherwise, None is returned.
    """

    if not scenario["scenario"]["attributesSimulation"]["useFixedSeed"]:
        return None

    return scenario["scenario"]["attributesSimulation"]["fixedSeed"]


def find_target_at_position(targets: list, coordinates: list) -> list:
    """Returns the IDs of the targets at the specified positions.

    Parameters:
    -----------
    targets : list
        A list of dictionaries of the targets.
    coordinates : list
        The coordinates of a points in the targets' areas.

    Returns:
    --------
    int
        The IDs of all targets which include at least of the coordinates.
    """

    ids = []

    for target in targets:
        for coordinate in coordinates:
            if is_inside(coordinate, target["shape"]):
                ids.append(target["id"]) if target["id"] not in ids else ids

    return ids


def get_next_free_id(scenario: dict) -> int:
    """Returns the first unassigned ID of the scenario.
    This function returns the highest existing ID + 1.
    Potentially missing IDs are not returned.

    Parameters:
    -----------
    scenario : dict
        The dictionary of the scenario.

    Returns:
    --------
    int
        The first unassigned ID.
    """

    used_ids = []

    # Store all IDs used by any scenario element.
    for obstacle in scenario["scenario"]["topography"]["obstacles"]:
        if "id" in obstacle:
            used_ids.append(obstacle["id"])

    for measurementArea in scenario["scenario"]["topography"]["measurementAreas"]:
        if "id" in measurementArea:
            used_ids.append(measurementArea["id"])

    for stair in scenario["scenario"]["topography"]["stairs"]:
        if "id" in stair:
            used_ids.append(stair["id"])

    for target in scenario["scenario"]["topography"]["targets"]:
        if "id" in target:
            used_ids.append(target["id"])

    for targetChanger in scenario["scenario"]["topography"]["targetChangers"]:
        if "id" in targetChanger:
            used_ids.append(targetChanger["id"])

    for absorbingArea in scenario["scenario"]["topography"]["absorbingAreas"]:
        if "id" in absorbingArea:
            used_ids.append(absorbingArea["id"])

    for aerosolCloud in scenario["scenario"]["topography"]["aerosolClouds"]:
        if "id" in aerosolCloud:
            used_ids.append(aerosolCloud["id"])

    for droplet in scenario["scenario"]["topography"]["droplets"]:
        if "id" in droplet:
            used_ids.append(droplet["id"])

    for source in scenario["scenario"]["topography"]["sources"]:
        if "id" in source:
            used_ids.append(source["id"])

    for dynamicElement in scenario["scenario"]["topography"]["dynamicElements"]:
        if "id" in dynamicElement["attributes"]:
            used_ids.append(dynamicElement["attributes"]["id"])

    return max(used_ids) + 1


def verify_coordinates(scenario: dict, coordinates: tuple) -> bool:
    """Verify whether a pedestrian can be placed in the given coordinates.

    Parameters:
    -----------
    scenario : dict
        The dictionary of the scenario.
    coordinates : tuple
        The coordinates where the pedestrian should be placed.

    Returns:
    --------
    bool
        True if the coordinates are inside the simulation area and are not
        conflicting with an obstacle. False otherwise.
    """

    # Out of bounds check
    if (
        scenario["scenario"]["topography"]["attributes"]["bounded"]
        and not (
            scenario["scenario"]["topography"]["attributes"]["bounds"]["x"]
            + scenario["scenario"]["topography"]["attributes"]["boundingBoxWidth"]
            <= coordinates[0] <=
            scenario["scenario"]["topography"]["attributes"]["bounds"]["x"]
            + scenario["scenario"]["topography"]["attributes"]["bounds"]["width"]
            - scenario["scenario"]["topography"]["attributes"]["boundingBoxWidth"]
            and
            scenario["scenario"]["topography"]["attributes"]["bounds"]["y"]
            + scenario["scenario"]["topography"]["attributes"]["boundingBoxWidth"]
            <= coordinates[1] <=
            scenario["scenario"]["topography"]["attributes"]["bounds"]["y"]
            + scenario["scenario"]["topography"]["attributes"]["bounds"]["height"]
            - scenario["scenario"]["topography"]["attributes"]["boundingBoxWidth"]
        )
    ):
        return False

    # Obstacle check
    for obstacle in scenario["scenario"]["topography"]["obstacles"]:
        if is_inside(coordinates, obstacle["shape"]):
            return False

    return True


def create_pedestrian(
        coordinates: tuple, pedestrian_id: int, targets: list,
        attributes: dict, seed: int | None
) -> dict:
    """Creates the dictionary of a new pedestrian.

    Parameters:
    -----------
    coordinates : tuple
        The starting position of the pedestrian.
    pedestrian_id : int
        The ID which will be assigned to the pedestrian
    targets : list
        A list of the targets' IDs the pedestrian is trying to reach.
    attributes : list
        The attributes of pedestrians.
    seed : int | None
        The seed used for the random sampling of the pedestrian's speed
        if one is used. Otherwise, None.

    Returns:
    --------
    dict
        The dictionary of the new pedestrian.
        Most fields are filled with default values.
    """

    pedestrian = {}

    attributes["id"] = pedestrian_id
    pedestrian["attributes"] = attributes

    pedestrian["source"] = None
    pedestrian["targetIds"] = targets

    pedestrian["nextTargetListIndex"] = 0
    pedestrian["isCurrentTargetAnAgent"] = False

    pedestrian["position"] = {"x": coordinates[0], "y": coordinates[1]}
    pedestrian["velocity"] = {"x": 0.0, "y": 0.0}

    # Sample the free flow speed from a Gaussian distribution until it is in the
    # desired speed range
    if seed:
        random.seed(seed)

    free_flow_speed = random.gauss(
        attributes["speedDistributionMean"],
        attributes["speedDistributionStandardDeviation"]
    )

    while not (
        attributes["minimumSpeed"] <= free_flow_speed <= attributes["maximumSpeed"]
    ):
        free_flow_speed = random.gauss(
            attributes["speedDistributionMean"],
            attributes["speedDistributionStandardDeviation"]
        )

    pedestrian["freeFlowSpeed"] = free_flow_speed

    pedestrian["followers"] = []
    pedestrian["idAsTarget"] = -1
    pedestrian["isChild"] = False
    pedestrian["isLikelyInjured"] = False

    pedestrian["psychologyStatus"] = {
        "mostImportantStimulus": None,
        "threatMemory": {
            "allThreats": [],
            "latestThreatUnhandled": False
        },
        "selfCategory": "TARGET_ORIENTED",
        "groupMembership": "OUT_GROUP",
        "knowledgeBase": {
            "knowledge": [],
            "informationState": "NO_INFORMATION"
        },
        "perceivedStimuli": [],
        "nextPerceivedStimuli": []
    }
    pedestrian["healthStatus"] = None
    pedestrian["infectionStatus"] = None

    pedestrian["groupIds"] = []
    pedestrian["groupSizes"] = []
    pedestrian["agentsInGroup"] = []

    pedestrian["trajectory"] = {"footSteps": []}
    pedestrian["modelPedestrianMap"] = None

    pedestrian["type"] = "PEDESTRIAN"

    return pedestrian


def add_pedestrian(pedestrian: dict, scenario: dict) -> dict:
    """Add a pedestrian to a scenario.

    Parameters:
    -----------
    pedestrian : dict
        The dictionary of the pedestrian.
    scenario : dict
        The dictionary of the scenario.

    Returns:
    --------
    dict
        The scenario including the pedestrian.
    """

    scenario["scenario"]["topography"]["dynamicElements"].append(
        pedestrian
    )

    return scenario


def adjust_name(new_name: str | None, scenario: dict) -> dict:
    """Adjust the name of a scenario.

    Parameters:
    -----------
    new_name : str | None
        The new name of the scenario. If None is passed, the new
        name will be the old name followed by "_modified".
    scenario : dict
        The dictionary of the scenario.

    Returns:
    --------
    dict
        The dictionary of the scenario with the adjusted name.
    """

    if new_name and scenario["name"] != new_name:
        scenario["name"] = new_name
    else:
        scenario["name"] += "_modified"

    return scenario


def is_inside(coordinates: tuple, shape: dict) -> bool:
    """Test whether a point is inside a scenario element.

    Parameters:
    -----------
    coordinates : tuple
        The coordinates of the point.
    shape : dict
        The dictionary of the element's shape.

    Returns:
    --------
    bool
        True if the point is inside the element, otherwise false.
    """

    # Rectangles and polygons require different approaches
    if shape["type"] == "RECTANGLE":
        return (
            shape["x"] <= coordinates[0] <= shape["x"] + shape["width"]
            and shape["y"] <= coordinates[1] <= shape["y"] + shape["height"]
        )

    # The check whether a point is inside a polygon below is adapted from
    # https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
    inside = False

    # Store the first point in the polygon and initialize the second point
    p1 = shape["points"][len(shape["points"]) - 1]

    # Loop through each edge in the polygon
    for i in range(0, len(shape["points"])):
        # Get the next point in the polygon
        p2 = shape["points"][i]

        # Check if the point is on the same height as one point of the edge
        if min(p1["y"], p2["y"]) < coordinates[1] <= max(p1["y"], p2["y"]):
            # Check if the point is to the left of the maximum x coordinate of the edge
            if coordinates[0] <= max(p1["x"], p2["x"]):
                # Calculate the x value of the intersection of the horizontal
                # line connecting the point to the edge
                x_intersection = (
                        (coordinates[1] - p1["y"]) * (p2["x"] - p1["x"]) / (p2["y"] - p1["y"]) + p1["x"]
                )

                # Check if the point is on the same line as the edge or to the left of the intersection
                if p1["x"] == p2["x"] or coordinates[0] <= x_intersection:
                    # Flip the inside flag
                    inside = not inside

        # Store the current point as the first point for the next iteration
        p1 = p2

    # Return the value of the inside flag
    return inside

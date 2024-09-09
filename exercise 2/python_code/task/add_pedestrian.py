import numpy as np
import csv
import json
import os

def update_scenario(root, input_file, output_file, pedestrian):
    with open(root + input_file + '.scenario', 'r') as infile:
        data = json.loads(infile.read())
    data['name'] = output_file
    data['scenario']['topography']['dynamicElements'].append(ped)
    with open(root + output_file + '.scenario', 'w') as outfile:
        outfile.write(json.dumps(data, indent=2))




ped = {
    "source" : None,
"targetIds" : [ 10 ],
"nextTargetListIndex" : 0,
"isCurrentTargetAnAgent" : False,
"position" : {
    "x" : 17,
    "y" : 3.0
},
"velocity" : {
    "x" : 0.0,
    "y" : 0.0
},
"freeFlowSpeed" : 0.7547668838665426,
"followers" : [ ],
"idAsTarget" : -1,
"isChild" : False,
"isLikelyInjured" : False,
"psychologyStatus" : {
    "mostImportantStimulus" : None,
    "threatMemory" : {
    "allThreats" : [ ],
    "latestThreatUnhandled" : False
    },
    "selfCategory" : "TARGET_ORIENTED",
    "groupMembership" : "OUT_GROUP",
    "knowledgeBase" : {
    "knowledge" : [ ],
    "informationState" : "NO_INFORMATION"
    },
    "perceivedStimuli" : [ ],
    "nextPerceivedStimuli" : [ ]
},
"healthStatus" : None,
"infectionStatus" : None,
"groupIds" : [ ],
"groupSizes" : [ ],
"agentsInGroup" : [ ],
"trajectory" : {
    "footSteps" : [ ]
},
"modelPedestrianMap" : None,
"type" : "PEDESTRIAN"
}

root = "./scenarios/"

update_scenario(root, "Corner", "Task3", ped)

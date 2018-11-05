import time
import numpy as np

from wrapper import get_areas_wrapper

def get_areas(trigs, vertices):
    areas = np.zeros(len(trigs))
    for idx, trig in enumerate(trigs):
        trig_vertices = [vertices[t] for t in trig]
        v1 = trig_vertices[1] - trig_vertices[0]
        v2 = trig_vertices[2] - trig_vertices[0]
	areas[idx] = np.linalg.norm(np.cross(v1, v2)) / 2
    return areas

NUM_VERTICES = 10000
NUM_TRIGS = 100000

vertices = np.random.rand(NUM_VERTICES, 3)
trigs = np.random.randint(0, NUM_VERTICES, size=(NUM_TRIGS, 3))

start_time = time.time()
areas_c = get_areas_wrapper(trigs, vertices)
print("Time taken (C): %f s" % (time.time() - start_time))

start_time = time.time()
areas_python = get_areas(trigs, vertices)
print("Time taken (Python): %f s" % (time.time() - start_time))

assert(np.all(np.isclose(areas_c, areas_python)))


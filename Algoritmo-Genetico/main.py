from genetic_algorithm import horary_genetic_algorithm as hga
from genetic_algorithm import init_random_collection as irc
import matplotlib.pyplot as plt

agh = hga(num_asignaturas=5, num_horas=8, size_collection=50, gen=200)

rand = irc(prob_indisponibilidad=0.1)
C1 = rand.create_collection()
C2 = rand.create_collection()
ind = rand.unavailability(C1)


result = agh.fit(C1, C2, ind)

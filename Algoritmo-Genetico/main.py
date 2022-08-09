#import genetic_algorithm as GA
from genetic_algorithm import horary_genetic_algorithm as hga
from genetic_algorithm import init_random_collection as irc
import matplotlib.pyplot as plt

#agh = GA.horary_genetic_algorithm(num_asignaturas=5, num_horas=6, size_collection=50, horas_total=30, gen=200)
agh = hga(num_asignaturas=5, num_horas=8, size_collection=50, gen=200)

rand = irc(prob_indisponibilidad=0.1)
C1 = rand.create_collection()
C2 = rand.create_collection()
ind = rand.unavailability(C1)


result = agh.fit(C1, C2, ind)

# INTERACCION DEL FIT MANUAL
gen = 50
evo_FC1 = []
evo_FC2 = []
max_FC1 = []
max_FC2 = []
min_FC1 = []
min_FC2 = []
for i in range(gen):

        
    h1c1 = agh.hard_1(C1, ind, 10)
    h1c2 = agh.hard_1(C2, ind, 10)
    h2 = agh.hard_2(C1, C2, 10)
    h3c1 = agh.hard_3(C1, 10)
    h3c2 = agh.hard_3(C2, 10)
    s1c1 = agh.soft_1(C1, 1)
    s1c2 = agh.soft_1(C2, 1)
    s2c1 = agh.soft_2(C1, 1)
    s2c2 = agh.soft_2(C2, 1)
    
    pc1 = agh.score(C1, h1c1, h2, h3c1, s1c1, s2c1)
    pc2 = agh.score(C2, h1c2, h2, h3c2, s1c2, s2c2)
    
    fc1, probC1 = agh.fitness(C1, pc1)
    fc2, probC2 = agh.fitness(C2, pc2)
    
    C1 = agh.cross(C1, probC1)
    C2 = agh.cross(C2, probC1)
    
    C1 = agh.mutation(C1, 0.1)
    C2 = agh.mutation(C2, 0.1)

    media_f1 = sum(fc1)/len(C1)
    media_f2 = sum(fc2)/len(C2)
    evo_FC1.append(media_f1)
    evo_FC2.append(media_f2)
    max_FC1.append(max(fc1))
    max_FC2.append(max(fc2))
    min_FC1.append(min(fc1))
    min_FC2.append(min(fc2))
    
x = list(range(gen))
plt.plot(x, evo_FC1, color = 'r')
plt.plot(x, max_FC1, color = 'r', linestyle = '--')
plt.plot(x, min_FC1, color = 'r', linestyle = '--')
plt.title('Evolucion de la Fitness - Clase 1')
plt.xlabel('Generaciones')
plt.ylabel('Fitness')
plt.show()
plt.plot(x, evo_FC2, color = 'b')
plt.plot(x, max_FC2, color = 'b', linestyle = '--')
plt.plot(x, min_FC2, color = 'b', linestyle = '--')
plt.title('Evolucion de la Fitness - Clase 2')
plt.xlabel('Generaciones')
plt.ylabel('Fitness')
plt.show()


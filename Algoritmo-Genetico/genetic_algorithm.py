import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class horary_genetic_algorithm():
    
    def __init__(self, num_asignaturas=5, num_horas=6, size_collection=50, gen=20):
        self.num_asignaturas = num_asignaturas
        self.num_horas = num_horas
        self.size_collection = size_collection
        self.horas_total = num_asignaturas * num_horas
        self.gen = gen
    
    # Condiciones para la evolucion del algoritmo 
    # hard
    def hard_1(self, coleccion_horarios_clase, indisponibilidad, importancia_H):
        ajuste_H1_clase = []
        H1 = []
        for i in range(len(coleccion_horarios_clase)):
            h1 = coleccion_horarios_clase[i]
            ajuste = []
            for p in range(len(h1)):
                prof = indisponibilidad[h1[p]] 
                if prof[p] == 0:
                    ajuste.append(0)
                else:
                    ajuste.append(importancia_H)
            H1.append(ajuste)
            ajuste_H1_clase.append(sum(H1[i]))
        return ajuste_H1_clase


    def hard_2(self, coleccion_horarios_clase1, coleccion_horarios_clase2, importancia_H):
        ajuste_H2_clase1 = []
        for i in range(len(coleccion_horarios_clase1)):
            aux = 0
            for j in range(len(coleccion_horarios_clase1[1])):
                if coleccion_horarios_clase1[i][j] == coleccion_horarios_clase2[i][j]:
                    aux = aux + importancia_H
            ajuste_H2_clase1.append(aux)
        return ajuste_H2_clase1


    def hard_3(self, coleccion_horarios_clase, importancia_H):
        ajuste_H3_clase = []
        for i in range(len(coleccion_horarios_clase)):
            a = Counter(coleccion_horarios_clase[i])
            pool = range(self.num_asignaturas)
            aux = 0
            for i in range(len(pool)+1):
                if a[i] != self.num_horas:
                    aux = aux + importancia_H
            ajuste_H3_clase.append(aux)  
        return ajuste_H3_clase
    
    
    def soft_1(self, coleccion_horarios_clase, importancia_S):
        n = int(len(coleccion_horarios_clase[1])/5)
        ajuste_S1_clase = []
        d = []
        pool = range(self.num_asignaturas)
        for i in range(len(coleccion_horarios_clase)):
            h = coleccion_horarios_clase[i]
            days = [h[i:i+n] for i in range(0, len(h), n)] #Divide en partes de 6 elementos cada vector horario.
            for i in range(len(days)):
                a = Counter(days[i])
                aux = 0
                for i in range(len(pool)+1):
                    if a[i] > 2:
                        aux = aux + importancia_S
                d.append(aux)
            ajuste_S1_clase.append(sum(d))
            d = []
        return ajuste_S1_clase
    
    
    def soft_2(self, coleccion_horarios_clase, importancia_S):
        n = int(len(coleccion_horarios_clase[1])/5)
        ajuste_S2_clase = []
        pool = range(self.num_asignaturas)
        for i in range(len(coleccion_horarios_clase)):
            h = coleccion_horarios_clase[i]
            days = [h[i:i+n] for i in range(0, len(h), n)] #Divide en partes de 6 elementos cada vector horario.
            aux = 0
            for i in range(len(days)):
                days[i] = np.array(days[i])    
                a = Counter(days[i])
                pos = []
                for t in range(len(pool)+1):
                    if a[t] > 1:
                        z = np.where(days[i] == t)
                        pos.append(int(z[0][0]))
                for j in range(len(pos)):
                    if days[i][pos[j]] != days[i][pos[j]+1]: 
                        aux = aux + importancia_S
            ajuste_S2_clase.append(aux)
        return ajuste_S2_clase
         
    
    def score(self, coleccion_horarios_clase, ajuste_H1_clase, ajuste_H2_clase, ajuste_H3_clase, ajuste_S1_clase, ajuste_S2_clase):
        cond_clase = []
        for i in range(len(coleccion_horarios_clase)):
            cond_clase.append(ajuste_H1_clase[i] + ajuste_H2_clase[i] + ajuste_H3_clase[i] + ajuste_S1_clase[i] + ajuste_S2_clase[i])
        return cond_clase 
    
    
    def fitness(self, coleccion_horarios_clase, cond_clase):
        fitness_clase = []
        prob_selec_clase = []
        for i in range(len(coleccion_horarios_clase)):
            suma = cond_clase[i]*(-1)
            fitness_clase.append(suma)
        aux = []
        minimo = min(fitness_clase)
        for i in range(len(fitness_clase)):
            aux.append(minimo - 1 - fitness_clase[i])
        for i in range(len(fitness_clase)):
            prob_selec_clase.append((aux[i]/np.array(aux).sum()))
        return fitness_clase, prob_selec_clase
    
    
    def cross(self, coleccion_horarios_clase, prob_selec_clase):
        size_poblacion = len(coleccion_horarios_clase)
        soon_clase = []
        for i in range(size_poblacion//2):
            parents = np.random.choice(size_poblacion, 2, p=prob_selec_clase)
            cross_point = np.random.randint(len(coleccion_horarios_clase[0]))
            soon_clase.append(coleccion_horarios_clase[parents[0]][:cross_point] +
                        coleccion_horarios_clase[parents[1]][cross_point:])
            soon_clase.append(coleccion_horarios_clase[parents[1]][:cross_point] +
                        coleccion_horarios_clase[parents[0]][cross_point:])
        coleccion_horarios_clase = soon_clase
        return coleccion_horarios_clase
    
    
    def mutation(self, coleccion_horarios_clase, prob):
        pool = range(self.num_asignaturas)
        for i in range(len(coleccion_horarios_clase[0])):
            mutate_horario = coleccion_horarios_clase[i]
            for p in range(1, len(mutate_horario)):
                if np.random.random() < prob:
                    mutation = np.random.choice(pool)
                    mutate_horario = mutate_horario[0:p] + \
                        [mutation] + mutate_horario[p+1:]
            coleccion_horarios_clase[i] = mutate_horario
            return coleccion_horarios_clase
    
    
    def fit(self, C1, C2, ind, gen=20, ih1=10, ih2=10, ih3=10, is1=2, is2=2, prob_mutacion=0.1):
        
        evo_FC1 = []
        evo_FC2 = []
        max_FC1 = []
        max_FC2 = []
        min_FC1 = []
        min_FC2 = []
        for i in range(self.gen):
            H1C1 = self.hard_1(C1, ind, ih1)
            H1C2 = self.hard_1(C2, ind, ih1)
            H2 = self.hard_2(C1, C2, ih2)
            H3C1 = self.hard_3(C1, ih3)
            H3C2 = self.hard_3(C2, ih3)
            S1C1 = self.soft_1(C1, is1)
            S1C2 = self.soft_1(C2, is1)
            S2C1 = self.soft_2(C1, is2)
            S2C2 = self.soft_2(C2, is2)
            #PENALIZACIONES
            PC1 = self.score(C1, H1C1, H2, H3C1, S1C1, S2C1)
            PC2 = self.score(C2, H1C2, H2, H3C2, S1C2, S2C2)
            
            #FITNESS
            FC1, ProbC1 = self.fitness(C1, PC1)
            FC2, ProbC2 = self.fitness(C2, PC2)
            
            #ENTRECRUZAMIENTO
            C1 = self.cross(C1, ProbC1)
            C2 = self.cross(C2, ProbC2)
            
            #MUTACION
            C1 = self.mutation(C1, prob_mutacion)
            C2 = self.mutation(C2, prob_mutacion)
            
            #GUARDAR RESULTADOS PARA EVALUAR RENDIMIENTO DEL MODELO
            media_f1 = sum(FC1)/self.size_collection
            media_f2 = sum(FC2)/self.size_collection
            evo_FC1.append(media_f1)
            evo_FC2.append(media_f2)
            max_FC1.append(max(FC1))
            max_FC2.append(max(FC2))
            min_FC1.append(min(FC1))
            min_FC2.append(min(FC2))
    
        #GUARDAR EN LA VARIABLE SOLUCION CUANDO SE GENERE LA SOLUCION OPTIMA
        solucion_C1 = 0
        solucion_C2 = 0
        for i in range(len(FC1)):
            if FC1[i] == 0 and FC2[i] == 0:
                solucion_C1 = self.C1[i]
                solucion_C2 = self.C2[i]
    
        x = list(range(self.gen))
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
    
        return ((C1, C2),(solucion_C1, solucion_C2))
     
           
class init_random_collection():
    
    def __init__(self, prob_indisponibilidad=0.1):
        horary_genetic_algorithm.__init__(self)
        self.prob_indisponibilidad = prob_indisponibilidad
      
    # Se crea una coleccion de horarios de forma totalmente aleatoria
    def create_collection(self):
        coleccion_horarios_clase = []
        pool = range(self.num_asignaturas)
        for i in range(self.size_collection):
            horario_clase = list(np.random.choice(pool, self.horas_total))
            coleccion_horarios_clase.append(horario_clase)
        return coleccion_horarios_clase
    
    # Se crea un vector de indisponibilidades para los profesores
    def unavailability(self, coleccion_horarios_clase):
        indisponibilidad = []
        pool = range(self.num_asignaturas)
        for i in range(len(pool)):
            prof = []
            for i in range(len(coleccion_horarios_clase[i])):
                if np.random.random() < self.prob_indisponibilidad:
                    prof.append(1)
                else:
                    prof.append(0)
            indisponibilidad.append(prof)
        return indisponibilidad
        
    


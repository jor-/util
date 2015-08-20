import random
import numpy as np

import deap.creator
import deap.base
import deap.tools
import deap.algorithms

import util.logging
logger = util.logging.logger





def init_individual_uniformly(bounds):
    bounds = np.asanyarray(bounds)
    individual_len = len(bounds)
    individual = np.empty(individual_len)
    for i in range(individual_len):
        individual[i] = random.uniform(bounds[i,0], bounds[i,1])
    return individual


def minimize(f, bounds, number_of_initial_individuals=100, number_of_generations=50, probability_of_crossover=0.5, probability_of_mutating=0.2, probability_of_mutating_in_each_dimension=0.5, crowding_degree_of_crossover=0.1, crowding_degree_of_mutation=0.1, tournament_size=3):
    logger.debug('Minimize function with {} initial individuals and {} generations.'.format(number_of_initial_individuals, number_of_generations))

    ## configure individual
    deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    ## configure population
    toolbox = deap.base.Toolbox()
    toolbox.register("init_individual_uniformly", init_individual_uniformly, bounds)
    toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, toolbox.init_individual_uniformly)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    ## configure ea methods
    toolbox.register("evaluate", lambda x: (f(x),))
    toolbox.register("mate", deap.tools.cxSimulatedBinaryBounded, eta=crowding_degree_of_crossover, low=bounds[:,0].tolist(), up=bounds[:,1].tolist())
    toolbox.register("mutate", deap.tools.mutPolynomialBounded, eta=crowding_degree_of_mutation, low=bounds[:,0].tolist(), up=bounds[:,1].tolist(), indpb=probability_of_mutating_in_each_dimension)
    toolbox.register("select", deap.tools.selTournament, tournsize=tournament_size)

    ## configure stats
    stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    ## run optimization
    pop = toolbox.population(n=number_of_initial_individuals)
    hof = deap.tools.HallOfFame(1, similar=np.array_equal)
    pop, log = deap.algorithms.eaSimple(pop, toolbox, cxpb=probability_of_crossover, mutpb=probability_of_mutating, ngen=number_of_generations, stats=stats, halloffame=hof, verbose=True)
#     return pop, log, hof
    return hof[0]




# deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
# deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)
# toolbox = deap.base.Toolbox()
#
#
#
# # Attribute generator
# # toolbox.register("init_individual_uniformly", lambda :init_individual_uniformly(bounds))
# toolbox.register("init_individual_uniformly", init_individual_uniformly, bounds)
# # Structure initializers
# # toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.init_individual_uniformly, 100)
# toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, toolbox.init_individual_uniformly)
# toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
#
#
# toolbox.register("evaluate", evalOneMax)
# #toolbox.register("mate", tools.cxTwoPoint)
# # toolbox.register("mate", tools.cxTwoPointCopy)
# # toolbox.register("mate", deap.tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low.tolist(), up.tolist()))
# toolbox.register("mate", deap.tools.cxSimulatedBinaryBounded, eta=0.1, low=bounds[:,0].tolist(), up=bounds[:,1].tolist())
# # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# toolbox.register("mutate", deap.tools.mutPolynomialBounded, eta=0.1, low=bounds[:,0].tolist(), up=bounds[:,1].tolist(), indpb=0.05)
# # toolbox.register("select", deap.tools.selBest, k=20)
# toolbox.register("select", deap.tools.selTournament, tournsize=3)
#
# def main():
#     random.seed(64)
#     pop = toolbox.population(n=300)
#     #hof = tools.HallOfFame(1)
#     hof = deap.tools.HallOfFame(1, similar=np.array_equal)
#     stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#     pop, log = deap.algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
#     return pop, log, hof
#
# if __name__ == "__main__":
#     main()
#
# if __name__ == "__main__":
#     random.seed(64)
#     bounds = np.array([[0,1],[4,9]])
#     def evalOneMax(individual):
#         return (sum(individual),)
#     x = minimize(evalOneMax, bounds)
#     print(x)


"""
Chapter 5 from Classic Computer Science Problems In Python
https://github.com/davecom/ClassicComputerScienceProblemsInPython
"""

from abc import ABC, abstractmethod
from enum import Enum
from random import choices, random
from heapq import nlargest
from statistics import mean
from random import randrange, random
from copy import deepcopy


class Chromosome(ABC):
    @abstractmethod
    def fitness(self):
        ...

    @classmethod
    @abstractmethod
    def random_instance(cls):
        ...

    @abstractmethod
    def crossover(self, other):
        ...

    @abstractmethod
    def mutate(self):
        return None


class GeneticAlgorithm():
    selection_type = Enum("selection_type", ["ROULETTE", "TOURNAMENT"])
    def __init__(self, initial_population, threshold, max_generations=100,
                 mutation_chance=0.01, crossover_chance=0.7,
                 selection_type=selection_type.TOURNAMENT):
        self._population = initial_population
        self._threshold = threshold
        self._max_generations = max_generations
        self._mutation_chance = mutation_chance
        self._crossover_chance = crossover_chance
        self._selection_type = selection_type
        self._fitness = type(self._population[0]).fitness

    def _pick_roulette(self, wheel):
        return tuple(choices(self._population, weights=wheel, k=2))

    def _pick_tournament(self, num_participants):
        participants = choices(self._population, k=num_participants)
        return tuple(nlargest(2, participants, key=self._fitness))

    def _reproduce_and_replace(self):
        new_population = []

        # keep going until we've filled the new generation
        while len(new_population) < len(self._population):
            # pick the 2 parents
            if self._selection_type == GeneticAlgorithm.selection_type.ROULETTE:
                parents = self._pick_roulette([chromosome.fitness()
                                               for chromosome in self._population])
            else:
                parents = self._pick_tournament(len(self._population) // 2)

            # potentially crossover the 2 parents
            if random() < self._crossover_chance:
                p1, p2 = parents
                new_population.extend(p1.crossover(p2))
            else:
                new_population.extend(parents)

        # if we had an odd number, we'll have 1 extra, so we remove it
        self._population = new_population[:len(self._population)]

    def _mutate(self) -> None:
        # With _mutation_chance probability mutate each individual
        for individual in self._population:
            if random() < self._mutation_chance:
                individual.mutate()

    def run(self):
        """Run the genetic algorithm for max_generations iterations and return the best individual found"""
        best = max(self._population, key=self._fitness)

        for generation in range(self._max_generations):
            # early exit if we beat threshold
            if best.fitness() >= self._threshold:
                return best

            print(f"Generation {generation} Best {best.fitness()} Avg {mean(map(self._fitness, self._population))}")
            self._reproduce_and_replace()
            self._mutate()

            highest = max(self._population, key=self._fitness)
            if highest.fitness() > best.fitness():
                best = highest # found a new best
        return best # best we found in _max_generations


# DEMO
class SimpleEquation(Chromosome):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def fitness(self): # 6x - x^2 + 4y - y^2
        x, y = self.x, self.y  # solution: x=3, y=2
        return 6*x - x**2 + 4*y - y**2

    @classmethod
    def random_instance(cls):
        return cls(randrange(100), randrange(100))

    def crossover(self, other):
        child1, child2 = deepcopy(self), deepcopy(other)
        child1.y = other.y
        child2.y = self.y
        return child1, child2

    def mutate(self):
        if random() > 0.5: # mutate x
            self.x += 1 if random() > 0.5 else -1
        else: # otherwise mutate y
            self.y += 1 if random() > 0.5 else -1

    def __str__(self) -> str:
        return f"X: {self.x} Y: {self.y} Fitness: {self.fitness()}"


if __name__ == "__main__":
    initial_population = [SimpleEquation.random_instance() for _ in range(20)]
    ga = GeneticAlgorithm(initial_population, threshold=13.0,
                          max_generations=100, mutation_chance=0.1,
                          crossover_chance=0.7)
    result = ga.run()
    print(result)

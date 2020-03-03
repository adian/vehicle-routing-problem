import random
from typing import List, Dict

from haversine import haversine

NO_INITIALISED_FLOAT = -1.0
CONFIG: "Config"


class City:
    __distance_table = {}

    def __init__(self, name: str, lat: float, lng: float, demand=0):
        self.name = name
        self.lat = lat
        self.lng = lng
        self.demand = demand

    def get_distance_to(self, destination: "City") -> int:
        origin_cords = (self.lat, self.lng)
        dest_cords = (destination.lat, destination.lng)

        cords_key = (
            origin_cords + dest_cords
            if origin_cords > dest_cords
            else dest_cords + origin_cords
        )

        if cords_key in City.__distance_table:
            return City.__distance_table[cords_key]
        else:
            dist = int(haversine(origin_cords, dest_cords))
            City.__distance_table[cords_key] = dist
            return dist


class Vehicle:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.route: List[City] = []
        self.loaded = 0
        self.__distance = NO_INITIALISED_FLOAT

    def can_delivery_to(self, city: City) -> bool:
        return self.capacity - self.loaded - city.demand >= 0

    def deliver_to(self, city: City):
        if self.can_delivery_to(city):
            self.route.append(city)
            self.loaded += city.demand
        else:
            current_capacity = self.capacity - self.loaded
            raise ValueError(f"Bigger demand than capacity({city.demand} > {current_capacity}")

    @property
    def distance(self) -> float:
        if self.__distance == NO_INITIALISED_FLOAT:
            self.__calculate_distance()
        return self.__distance

    def __calculate_distance(self):
        self.__distance = 0.0
        for i in range(1, len(self.route)):
            last_city: City = self.route[i - 1]
            this_city: City = self.route[i]
            self.__distance += last_city.get_distance_to(this_city)


class InvalidVehiclesError(ValueError):
    pass


class Individual:
    def __init__(self, vehicles: List[Vehicle]):
        self.vehicles = vehicles
        self.__keys = []
        self.__distance = NO_INITIALISED_FLOAT
        self.__fitness = NO_INITIALISED_FLOAT

    @property
    def distance(self):
        if self.__distance == NO_INITIALISED_FLOAT:
            self.__calculate_distance()
        return self.__distance

    def __calculate_distance(self):
        self.__distance = 0.0
        for vehicle in self.vehicles:
            self.__distance += vehicle.distance

    @property
    def fitness(self):
        if self.__fitness == NO_INITIALISED_FLOAT:
            self.__fitness = 1 / self.distance
        return self.__fitness

    @property
    def keys_without_base_city(self) -> List[str]:
        if len(self.__keys) == 0:
            self.__calculate_keys_without_base_city()
        return self.__keys

    def __calculate_keys_without_base_city(self):
        self.__keys = []
        for vehicle in self.vehicles:
            for city in vehicle.route:
                if city != CONFIG.base_city:
                    self.__keys.append(city.name)

    @staticmethod
    def create_form_keys(city_keys: List[str]) -> "Individual" or None:
        result_vehicles = []
        current_vehicle_index = 0
        vehicle = Individual.get_vehicle(current_vehicle_index, CONFIG.vehicles)
        for name in city_keys:
            city = CONFIG.city_dictionary[name]
            if vehicle.can_delivery_to(city):
                vehicle.deliver_to(city)
            else:
                result_vehicles.append(vehicle)
                current_vehicle_index += 1
                try:
                    vehicle = Individual.get_vehicle(
                        current_vehicle_index, Config.vehicles
                    )
                except InvalidVehiclesError:
                    return None
                vehicle.deliver_to(city)

        result_vehicles.append(vehicle)
        for vehicle in result_vehicles:
            Individual.append_base_city(vehicle)
        return Individual(result_vehicles)

    @staticmethod
    def append_base_city(vehicle: Vehicle):
        vehicle.route.insert(0, Config.base_city)
        vehicle.route.append(Config.base_city)

    @staticmethod
    def get_vehicle(current_vehicle_index: int, vehicles: List[Vehicle]) -> Vehicle:
        if current_vehicle_index >= len(vehicles):
            raise InvalidVehiclesError("No vehicle available!")
        else:
            this_vehicle = vehicles[current_vehicle_index]
            return Vehicle(this_vehicle.capacity)


class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @property
    def best_individual(self):
        self.__sort_population_by_fitness()
        return self.individuals[0]

    def next_generation(self):
        elites = self.__get_elites()
        new_population = Population(
            [individual.copy() for individual in self.individuals]
        )

        new_population.mate_population()
        new_population.mutate_population()
        new_population.individuals += elites

        return new_population

    def __get_elites(self) -> List[Individual]:
        self.__sort_population_by_fitness()
        elites = []
        for individual in self.individuals[: Config.number_of_elites]:
            elites.append(individual.copy())
        return elites

    def __sort_population_by_fitness(self) -> None:
        self.individuals.sort(key=lambda individual: individual.fitness, reverse=True)

    def mate_population(self):
        selected_individuals = self.select_individuals()
        result = []
        for parent_1, parent_2 in zip(
                selected_individuals[::2], selected_individuals[1::2]
        ):
            child_1, child_2 = Population.mate_parents(parent_1, parent_2)
            result.append(child_1)
            result.append(child_2)
        self.individuals = result

    def select_individuals(self) -> List[Individual]:
        self.__sort_population_by_fitness()
        selected = []
        fitness_sum = sum(individual.fitness for individual in self.individuals)

        for _ in range(0, len(self.individuals) - Config.number_of_elites):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for individual in self.individuals:
                current += individual.fitness
                if current > pick:
                    selected.append(individual)
                    break
        return selected

    @staticmethod
    def mate_parents(parent_1: Individual, parent_2: Individual):
        child_1 = None
        child_2 = None

        while child_1 is None or child_2 is None:
            size = min(len(parent_1.keys_without_base_city), len(parent_2.keys_without_base_city))
            index_1, index_2 = sorted(random.sample(range(size), 2))

            temp_genes_1 = parent_1.keys_without_base_city[index_1: index_2 + 1] + parent_2.keys_without_base_city
            temp_genes_2 = parent_2.keys_without_base_city[index_1: index_2 + 1] + parent_1.keys_without_base_city

            child_1 = Individual.create_form_keys(Population.distinct(temp_genes_1))
            child_2 = Individual.create_form_keys(Population.distinct(temp_genes_2))
        return child_1, child_2

    @staticmethod
    def distinct(genes: List[str]):
        result = []
        for gene in genes:
            if gene not in result:
                result.append(gene)
        return result

    def mutate_population(self):
        for index in range(len(self.individuals)):
            if random.random() < Config.mutation_rate:
                individual = None
                while individual is None:
                    keys = self.individuals[index].keys_without_base_city
                    key_1, key_2 = random.sample(range(len(keys)), 2)
                    keys[key_1], keys[key_2] = keys[key_2], keys[key_1]
                    individual = Individual.create_form_keys(keys)
                self.individuals[index] = individual


def create_first_generation():
    individuals = []
    individuals_to_do = Config.max_population_size
    while individuals_to_do != 0:
        sample_keys = random.sample(Config.city_keys, len(Config.city_keys))
        individual = Individual.create_form_keys(sample_keys)
        if individual is not None:
            individuals.append(individual)
            individuals_to_do -= 1

    return Population(individuals)


class Config:

    def __init__(self,  base_city: City, cities: List[City], vehicles: List[Vehicle] ):
        self.base_city = base_city
        self.cities = cities
        self.__vehicles = vehicles
        self.city_dictionary = {}
        self.city_keys = []
        self.create_city_dict_and_city_keys()

    def create_city_dict_and_city_keys(self):
        self.city_dictionary: Dict[str, City] = {self.base_city.name: self.base_city}
        self.city_keys: List[str] = []

        for city in self.cities:
            self.city_dictionary[city.name] = city
            self.city_keys.append(city.name)

    @property
    def vehicles(self):
        self.__vehicles.



def run_vrp(config: Config):
    CONFIG = config

    print("end")
# def run_vrp(
#         base_city: City,
#         cities: List[City],
#         vehicles: List[Vehicle],
#         max_population_size: int,
#         number_of_generations: int,
#         number_of_elites: int,
#         mutation_rate: float,
# ):
#     Config.base_city = base_city
#     Config.mutation_rate = mutation_rate
#     Config.number_of_elites = number_of_elites
#     Config.create_city_dict(base_city, cities)
#     Config.vehicles = vehicles
#     Config.max_population_size = max_population_size
#
#     first_generation = create_first_generation()
#     generations = [first_generation]
#     for i in range(2, number_of_generations + 1):
#         generation = generations[-1].next_generation()
#         generations.append(generation)
#
#     last_generation = generations[-1]
#     print_information(last_generation.best_individual)
#
#     distance = [a.best_individual.distance for a in generations]
#     plt.plot(distance)
#     plt.ylabel('distance')
#     plt.xlabel('generations')
#     plt.savefig("graph.png")
#
#     print("end")


def print_information(individual: Individual):
    print(f"Best distance is {individual.distance}")
    for vehicle in individual.vehicles:
        print(get_vehicle_route_as_str(vehicle))


def get_vehicle_route_as_str(vehicle: Vehicle):
    cities_names = [city.name for city in vehicle.route]
    string = " -> ".join(cities_names)
    string += f"; distance: {vehicle.distance}; loaded: {vehicle.loaded}"
    return string

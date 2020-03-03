from typing import List

from vrp import City, run_vrp, Vehicle

from src.vrp import Config

cities = [
    City("Białystok", lat=53.129047, lng=23.156064, demand=500),
    City("Bielsko-Biała", lat=49.817876, lng=19.046868, demand=50),
    City("Chrzanów", lat=50.134922, lng=19.398866, demand=400),
    City("Gdańsk", lat=54.351003, lng=18.648136, demand=200),
    City("Gdynia", lat=54.521283, lng=18.523211, demand=100),
    City("Gliwice", lat=50.295736, lng=18.671931, demand=40),
    City("Gromnik", lat=49.837343, lng=20.957883, demand=200),
    City("Katowice", lat=50.263497, lng=19.02779, demand=300),
    City("Kielce", lat=50.867455, lng=20.627313, demand=30),
    City("Krosno", lat=49.68297, lng=21.765495, demand=60),
    City("Krynica", lat=49.421366, lng=20.958607, demand=50),
    City("Lublin", lat=51.245112, lng=22.561018, demand=60),
    City("Łódź", lat=51.761735, lng=19.447614, demand=160),
    City("Malbork", lat=54.035892, lng=19.03728, demand=100),
    City("Nowy Targ", lat=49.478094, lng=20.028996, demand=120),
    City("Olsztyn", lat=53.77958, lng=20.475994, demand=300),
    City("Poznań", lat=52.406778, lng=16.925043, demand=100),
    City("Puławy", lat=51.415651, lng=21.97192, demand=200),
    City("Radom", lat=51.404703, lng=21.142296, demand=100),
    City("Rzeszów", lat=50.042402, lng=21.98845, demand=60),
    City("Sandomierz", lat=50.68114, lng=21.75282, demand=200),
    City("Szczecin", lat=53.429367, lng=14.554567, demand=150),
    City("Szczucin", lat=50.309958, lng=21.074637, demand=60),
    City("Szklarska Poręba", lat=50.830241, lng=15.529729, demand=50),
    City("Tarnów", lat=50.011058, lng=20.975528, demand=70),
    City("Warszawa", lat=52.230922, lng=21.006243, demand=200),
    City("Wieliczka", lat=49.987537, lng=20.059498, demand=90),
    City("Wrocław", lat=51.10519, lng=17.038536, demand=40),
    City("Zakopane", lat=49.297752, lng=19.952584, demand=200),
    City("Zamość", lat=50.719623, lng=23.253661, demand=300),
]

base_city = City("Kraków", lat=50.065385, lng=19.942361)

vehicles: List[Vehicle] = [
    Vehicle(1000),
    Vehicle(1000),
    Vehicle(1000),
    Vehicle(1000),
    Vehicle(1000),
]

config = Config(
    base_city=base_city,

)

# run_vrp(
#     base_city=base_city,
#     cities=cities,
#     vehicles=vehicles,
#     max_population_size=100,
#     number_of_generations=5000,
#     number_of_elites=10,
#     mutation_rate=0.1,
# )



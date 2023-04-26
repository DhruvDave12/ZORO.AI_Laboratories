import numpy as np

class AGE_GROUP:
    def __init__(self, lowerLimit, upperLimit):
        self.LOWER_LIMIT = lowerLimit
        self.UPPER_LIMIT = upperLimit

class ZORO_DATA:
    def __init__(self):
        self.COVID_BEDS = 1000
        self.NORMAL_BEDS = 1000
        self.ITERATOR = 0
        AVAILABLE_AGES = [AGE_GROUP(0, 18), AGE_GROUP(18, 40), AGE_GROUP(40, 65), AGE_GROUP(65, 100)]
        self.AGE_GROUPS = AVAILABLE_AGES
        self.HOSPITALISED_PERCENT = [0.5, 1, 2, 10]
        POPULATION_LOCAL = {}
        for ageGroup in AVAILABLE_AGES:
            POPULATION_LOCAL[ageGroup] = np.random.rand() * 50000
        self.POPULATION = POPULATION_LOCAL
        
import math
import random
from itertools import combinations_with_replacement
from typing import List, Generator

def generate_all_schedules(N: int, T: int) -> list[list[int]]:
    def generate(current_schedule: list[int], remaining_patients: int, remaining_slots: int):
        if remaining_slots == 0:
            if remaining_patients == 0:
                schedules.append(current_schedule)
            return
        
        for i in range(remaining_patients + 1):
            generate(current_schedule + [i], remaining_patients - i, remaining_slots - 1)
    
    pop_size = math.comb(N+T-1, N)
    print(f"Number of possible schedules: {pop_size}")
    schedules = []
    generate([], N, T)
    return schedules

def serialize_schedules(N: int, T: int) -> None:
    file_path=f"experiments/n{N}t{T}.pickle"
    
    # Start timing
    start_time = time.time()
    
    # Open a file for writing serialized schedules
    with open(file_path, 'wb') as f:
        for comb in combinations_with_replacement(range(T), N):
            schedule = np.bincount(comb, minlength=T).tolist()
            pickle.dump(schedule, f)
    
    # End timing
    end_time = time.time()
    
    print(f"Number of possible schedules: {math.comb(N+T-1, N)}")
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Schedules are serialized and saved to {file_path}")

def load_schedules(file_path: [str]) -> list[list[int]]:
    # Start timing
    start_time = time.time()
    
    schedules = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                schedules.append(pickle.load(f))
            except EOFError:
                break
        # End timing
    end_time = time.time()
    print(f"Loading time: {end_time - start_time} seconds")
    return schedules

def generate_random_schedule(N: int, T: int) -> list[int]:
  """
  A function to generate a random schedule for N patients and T slots
  """
  slots = random.choices(range(T), k = N)
  schedule = [0]*T
  for slot in slots:
    schedule[slot] += 1
  return(schedule)

def random_combination_with_replacement(T: int, N: int, num_samples: int) -> List[List[int]]:
    """
    Generate random samples from the set of combinations with replacement without generating the entire population.

    Parameters:
    T (int): The range of elements to choose from, i.e., the maximum value plus one.
    N (int): The number of elements in each combination.
    num_samples (int): The number of random samples to generate.

    Returns:
    List[List[int]]: A list containing the randomly generated combinations.
    """
    def index_to_combination(index: int, T: int, N: int) -> List[int]:
        """
        Convert a lexicographic index to a combination with replacement.

        Parameters:
        index (int): The lexicographic index of the combination.
        T (int): The range of elements to choose from, i.e., the maximum value plus one.
        N (int): The number of elements in each combination.

        Returns:
        List[int]: The combination corresponding to the given index.
        """
        combination = []
        current = index
        for i in range(N):
            for j in range(T):
                combs = math.comb(N - i + T - j - 2, T - j - 1)
                if current < combs:
                    combination.append(j)
                    break
                current -= combs
        return combination

    # Calculate total number of combinations
    total_combinations = math.comb(N + T - 1, N)
    print(f"Total number of combinations: {total_combinations}")

    samples = []
    for _ in range(num_samples):
        # Randomly select an index
        random_index = random.randint(0, total_combinations - 1)
        # Convert index to combination
        sample = index_to_combination(random_index, T, N)
        samples.append(sample)

    return samples
    
if __name__ == "__main__":
  
    # Example usage of random_combination_with_replacement()
    T = 18
    N = 12
    num_samples = 5
    
    samples = random_combination_with_replacement(T, N, num_samples)
    for sample in samples:
        print(sample)

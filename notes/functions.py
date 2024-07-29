def generate_all_schedules(N, T):
    import math
    def generate(current_schedule, remaining_patients, remaining_slots):
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

def serialize_schedules(N, T):
    from itertools import combinations_with_replacement
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

def load_schedules(file_path):
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

def generate_random_schedule(N, T):
  import random
  slots = random.choices(range(T), k = N)
  schedule = [0]*T
  for slot in slots:
    schedule[slot] += 1
  return(schedule)

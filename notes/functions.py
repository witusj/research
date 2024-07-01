def generate_all_schedules(N, T):
    def generate(current_schedule, remaining_patients, remaining_slots):
        if remaining_slots == 0:
            if remaining_patients == 0:
                schedules.append(current_schedule)
            return
        
        for i in range(remaining_patients + 1):
            generate(current_schedule + [i], remaining_patients - i, remaining_slots - 1)
    
    schedules = []
    generate([], N, T)
    return schedules

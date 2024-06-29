import logging
import copy
import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.stats import lognorm
from scipy import signal
import plotly.graph_objs as go
import plotly.offline as pyo
import unittest
from itertools import chain, combinations
import plotly.graph_objs as go
import copy

# """
# Function to calculate the convolution of two arrays.
# 
# Args:
#     a (numpy.ndarray): The first array to be convolved.
#     b (numpy.ndarray): The second array to be convolved.
# 
# Returns:
#     numpy.ndarray: The convolution of the two input arrays.
# """
def convolve(a, b):
    
    # Initialize an empty array to store the result.
    c = np.array([])
    
    # Compute the convolution of the two arrays.
    for i in range(len(a)):
        # Get subsets of array expanded to the right.
        a_sub = a[0:i + 1].copy()
        b_sub = b[0:i + 1].copy()
        # Reverse b.
        b_rev = b_sub[::-1]
        # Compute the dot product of a and b_rev.
        c = np.append(c, np.dot(a_sub, b_rev))
    
    for i in range(1,len(a)):
        # Get subsets of array collapse from the right.
        a_sub = a[i:].copy()
        b_sub = b[i:].copy()
        # Reverse b.
        b_rev = b_sub[::-1]
        # Compute the dot product of a and b_rev.
        c = np.append(c, np.dot(a_sub, b_rev))
        
    return c


# """
# Function to convolve a distribution with itself n times.
# 
# Args:
#     a (numpy.ndarray): The distribution to be convolved.
#     n (int): The number of times to convolve the distribution with itself.
# 
# Returns:
#     numpy.ndarray: The convolution of the input distribution with itself n times.
# """
def convolve_n(a, n):
        
    # Initialize an empty array to store the result.
    c = np.array([])
    
    # If n is 0, return an array of zeros with length equal to the length of a, except for the first element which is 1.
    if n == 0:
        c = np.array(np.zeros(len(a)), dtype=np.float64)
        c[0] = 1
        return c
    
    # Convolve the distribution with itself n times.
    for i in range(n):
        # If this is the first iteration, set c equal to a.
        if i == 0:
            c = a
        # Otherwise, convolve c with a.
        else:
            c = np.convolve(c, a)
            
    return c
# 
# """
# Function to create an array of zero arrays according to a given shape array.
# 
# Args:
#      num_zeros (numpy.ndarray): The shape array.
#      l (int): The length of the zeros array.
#  
# Returns:
#      numpy.ndarray: The convolution of the input distribution with itself n times.
# """

def zero_arrays(num_zeros, l):
    result = []
    for n in num_zeros:
        zeros = np.zeros(l)
        result.append([zeros] * n)
    return result

print(f'Zero arrays are: {zero_arrays(np.array([1, 0, 3]), 4)}')

def calc_distr_limit(l):
    return int(max(l+4*l**0.5, 100))
  
class TestConvolve(unittest.TestCase):
    
    def test_convolve(self):
        a = np.array([
            0.4456796414,
            0.160623141,
            0.137676978,
            0.1032577335])

        b = np.array([
            0.006737946999,
            0.033689735,
            0.08422433749,
            0.1403738958])

        expected_output = np.convolve(a, b)
        
        self.assertTrue(np.allclose(convolve(a, b), expected_output))
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [list(item) for item in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]

def get_v_star(t):
    # Create an initial vector 'u' of zeros with length 't'
    u = np.zeros(t, dtype=np.int64)
    # Set the first element of vector 'u' to -1
    u[0] = -1
    # Set the last element of vector 'u' to 1
    u[-1] = 1
    # Initialize the list 'v_star' with the initial vector 'u'
    v_star = [u]
    # Loop over the length of 'u' minus one times
    for i in range(len(u) - 1):
        # Append the last element of 'u' to the front of 'u'
        u = np.append(u[-1], u)
        # Remove the last element of 'u' to maintain the same length
        u = np.delete(u, -1)
        # Append the updated vector 'u' to the list 'v_star'
        v_star.append(u)
    # Convert the list of vectors 'v_star' into a NumPy array and return it
    logging.info(f'V_star = {v_star}')
    return(np.array(v_star))

# Example of function call:
# This will create a 4x4 matrix where each row is a cyclically shifted version of the first row

def generate_search_neighborhood(schedule):
  N = sum(schedule)
  T = len(schedule)
  logging.info(f'The schedule = {schedule}')
  
  # Generate a matrix 'v_star' using the 'get_v_star' function
  v_star = get_v_star(T)
  
  # Generate all possible non-empty subsets (powerset) of the set {0, 1, 2, ..., T-1}
  # 'ids' will be a list of tuples, where each tuple is a subset of indices
  ids = list(powerset(range(T)))

  logging.info(f'Neighborhood size = {len(ids)}')

  # Select the vectors from 'v_star' that correspond to the indices in each subset
  # 'sub_sets' will be a list of lists, where each inner list contains vectors from 'v_star'
  sub_sets = [v_star[i] for i in ids]

  # Sum the vectors within each subset and flatten the result to get a 1-D array
  # 'summed_sets' will be a list of 1-D numpy arrays, where each array is the sum of vectors
  summed_sets = [np.sum(sub_sets[i], axis=0).flatten() for i in range(len(sub_sets))]
  logging.info(f'Summed sets = {summed_sets}')
  
  neighborhood = np.array([schedule + summed_sets[i] for i in range(len(summed_sets))])
  logging.info(f'Neighborhood = {neighborhood}')
  
  # Create a mask for rows with negative values
  mask = ~np.any(neighborhood < 0, axis=1)

  # Filter out rows with negative values using the mask
  filtered_neighborhood = neighborhood[mask]
  logging.info(f'And the neighborhood is {filtered_neighborhood}')
  return filtered_neighborhood

def generate_small_search_neighborhood(schedule):
  N = sum(schedule)
  T = len(schedule)
  logging.info(f'The schedule = {schedule}')
  
  # Generate a matrix 'v_star' using the 'get_v_star' function
  v_star = get_v_star(T)
  
  neighborhood = np.array([schedule + v_star[i] for i in range(len(v_star))])
  
  # Create a mask for rows with negative values
  mask = ~np.any(neighborhood < 0, axis=1)

  # Filter out rows with negative values using the mask
  filtered_neighborhood = neighborhood[mask]
  logging.info(f'And the neighborhood is {filtered_neighborhood}')
  return filtered_neighborhood

def distribute_patients(n_patients, n_timeslots):
    # Create a list with all slots initially empty
    distribution = [0] * n_timeslots

    # Place patients in timeslots
    for i in range(n_patients):
        # Calculate the slot for each patient
        slot = round(i * n_timeslots / n_patients)
        distribution[slot] = 1

    return distribution

def plot_timeline(slots, title):
    # Create a figure
    fig = go.Figure()

    # Iterate over each timeslot and add a bar for occupied slots
    for i, slot in enumerate(slots):
        fig.add_trace(go.Bar(x=[i], y=[slot], width=0.8, marker_color='black'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Timeslots",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False
    )

    # Set y-axis range
    fig.update_yaxes(range=[0, np.max(slots)])

    # Show the figure
    fig.show()
    
def service_time_with_no_shows(s, q):
  # """
  # Function to adjust a distribution of service times for no-shows
  # 
  # Args:
  #     s (numpy.ndarray): An array with service times.
  #     q (double): The fraction of no-shows.
  # 
  # Returns:
  #     numpy.ndarray: The adjusted array of service times.
  # """
  
  s_adj = [(1 - q) * float(si) for si in s]
  s_adj[0] = s_adj[0] + q
  return(s_adj)

def add_lists(short_list, long_list):
  # """
  # This function takes in two lists and returns a new list where each element 
  # is the sum of the elements from the input lists at the corresponding position.
  # If the lists are of different lengths, the shorter list is extended with zeros 
  # to match the length of the longer list.
  # 
  # Parameters:
  # - short_list (list): The shorter list of numbers.
  # - long_list (list): The longer list of numbers.
  # 
  # Returns:
  # - list: A list containing the element-wise sum of the two input lists.
  # """
  
  # Extend the short lists to the length of the long list with zeros
  short_list.extend([0] * (len(long_list) - len(short_list)))
  
  # Sum the elements of the two lists element-wise
  result = [a + b for a, b in zip(short_list, long_list)]
  
  return result

def calculate_rolling_convolution(p_y, s):
  conv_list = s
  limit = len(p_y)
  v = [x * p_y[0] for x in conv_list]
  for i in range(1, limit):
    conv_list = np.convolve(conv_list, s)
    v = add_lists(v, [x * p_y[i] for x in conv_list])
    
  return(v)

def generate_schedules(N):
    """
    Generate all possible ways to distribute N patients into 3 timeslots.
    The function returns a list of lists, where each sublist represents
    a distribution of patients across the 3 timeslots such that the sum of
    the numbers in the sublist is N.

    Parameters:
    N (int): The total number of patients to be distributed.

    Returns:
    List[List[int]]: A list of lists where each sublist contains 3 integers
    representing a possible distribution of patients across the 3 timeslots.
    """
    # This list will hold the result
    schedules = []

    # Iterate over all possible values of 'a' from 0 to N (inclusive)
    for a in range(N + 1):
        # For each value of 'a', iterate over all possible values of 'b' from 0 to N-a (inclusive)
        for b in range(N - a + 1):
            # Compute 'c' such that the sum of 'a', 'b', and 'c' is N
            c = N - a - b
            # Append the combination [a, b, c] to the schedules list
            schedules.append([a, b, c])

    # Return the list of all possible schedules
    return schedules
  
class Schedule:
  def __init__(self, x, d, s, q, omega):
    
      # Clear the log file by opening it in write mode
      with open('logs.txt', 'w'):
            pass

        # Configure logging
      logging.basicConfig(filename='logs.txt', encoding='utf-8', level=logging.DEBUG)

      if not all(isinstance(i, np.integer) and i >= 0 for i in x):
          raise ValueError("All elements in x must be non-negative integers.")
      if not isinstance(d, int) or d <= 0:
          raise ValueError("d must be a positive integer.")
      if not all(isinstance(i, float) and 0 <= i <= 1 for i in s):
          raise ValueError("All elements in s must be floats between 0 and 1.")
      if not isinstance(q, float) and 0 <= q <= 1:
          raise ValueError("q must be a float between 0 and 1.")
      if not isinstance(omega, float) and 0 <= omega <= 1:
          raise ValueError("omegea must be a float between 0 and 1.")

      self.parameters = {'x': x, 'd': d, 's': s, 'q': q, 'omega': omega}
      self.parameters['s'] = service_time_with_no_shows(self.parameters['s'], self.parameters['q'])
      self._initialize_system()

  def _initialize_system(self):
      self.state = 0
      self.system = {
          'p_min': [],  # Initialize as empty list
          'p_plus': [],  # Initialize as empty list
          'w': [[] for _ in self.parameters['x']],  # Initialize as list of lists
          'ew': [],  # Initialize as empty list
          'loss': None,
          'search_path': [] # Initialize as empty list
      }
      self.system['p_min'].append(np.zeros(len(self.parameters['s']), dtype=np.float64))
      self.system['p_min'][0][0] = 1  # The first patient in the schedule has waiting time zero with probability 1
      if self.parameters['x'][0] > 0:  # Only calculate waiting times if there are patients scheduled in the state
          self.system['w'][0].append(self.system['p_min'][0].copy())
          for i in range(1, self.parameters['x'][0]):
              convolved = np.convolve(self.system['w'][0][i-1], self.parameters['s'])
              self.system['w'][0].append(convolved)
      self._update_p_plus(0)
      self.state = 1

  def _update_p_plus(self, state):
      if self.parameters['x'][state] == 0:
          self.system['p_plus'].append(self.system['p_min'][state].copy())
      else:
          convolved = np.convolve(self.system['w'][state][-1], self.parameters['s'], mode='full')
          self.system['p_plus'].append(convolved)
      logging.info(f"p_plus = {self.system['p_plus'][state]}")

  def _calculate_state(self):
      logging.info(f'{datetime.datetime.now()} - State = {self.state}')
      new_p_min_length = len(self.system['p_plus'][self.state-1]) - self.parameters['d']
      new_p_min = np.zeros(max(len(self.system['p_plus'][self.state-1]), new_p_min_length + self.parameters['d']), dtype=np.float64)
      new_p_min[0] = np.sum(self.system['p_plus'][self.state-1][:(self.parameters['d'] + 1)])
      new_p_min[1:new_p_min_length] = self.system['p_plus'][self.state-1][(self.parameters['d'] + 1):]
      self.system['p_min'].append(new_p_min)
      if self.parameters['x'][self.state] > 0:
          self.system['w'][self.state].append(self.system['p_min'][self.state].copy())
          logging.info(self.system['w'][self.state][0])
          for i in range(1, self.parameters['x'][self.state]):
              convolved = np.convolve(self.system['w'][self.state][i-1], self.parameters['s'])
              self.system['w'][self.state].append(convolved)
              logging.info(self.system['w'][self.state][i])
      self._update_p_plus(self.state)

  def calculate_system_states(self, until=1):
      while self.state < until:
          self._calculate_state()
          self.state += 1

  def calculate_tardiness(self):
      new_p_min_length = len(self.system['p_plus'][-1]) + self.parameters['d']
      new_p_min = np.zeros(new_p_min_length, dtype=np.float64)
      new_p_min[0] = np.sum(self.system['p_plus'][-1][:(self.parameters['d'] + 1)])
      new_p_min[1:len(self.system['p_plus'][-1]) - self.parameters['d']] = self.system['p_plus'][-1][(self.parameters['d'] + 1):]
      self.system['p_min'][-1] = new_p_min

  def calculate_wait_times(self):
      for interval, wtdists in enumerate(self.system['w']):
          ew = 0
          for nr, dist in enumerate(wtdists):
              a = range(len(dist))
              b = dist
              meanwt = np.dot(a, b)
              logging.info(f"Mean waiting time for patient {nr} in interval {interval} = {meanwt}")
              ew += meanwt
          self.system['ew'].append(ew)
          
  def calculate_loss(self):
      omega = self.parameters['omega']
      tot_wt = np.sum(self.system["ew"])
      self.calculate_tardiness()
      indices = np.arange(len(self.system['p_min'][-1])) 
      exp_tard = (indices * self.system['p_min'][-1]).sum()
      self.system['loss'] = omega * tot_wt / sum(self.parameters['x']) + (1 - omega) * exp_tard
          
  def local_search(self, omega=0.5):
      omega = self.parameters['omega']
      # Calculate initial loss
      test_wt = np.sum(self.system["ew"])  # Use np.sum instead of list's sum method
      self.calculate_tardiness()
      indices = np.arange(self.system['p_min'][-1].size) 
      exp_tard = (indices * self.system['p_min'][-1]).sum()
      lowest_loss = omega * test_wt / np.sum(self.parameters['x']) + (1 - omega) * exp_tard
      store_optim = {'x': self.parameters['x'].copy(), 'system': copy.deepcopy(self.system), 'tot_wt': test_wt}
  
      # Initialize search path
      search_path = []
  
      # Save initial schedule and loss
      search_path.append({'schedule': self.parameters['x'].copy(), 'loss': lowest_loss})
      
      # Continue the search until no improvement is found
      while True:  # Start an outer loop that will continue until explicitly broken
          nh = generate_search_neighborhood(self.parameters['x'])  # Generate a new neighborhood
          improved = False  # Flag to check if an improvement was found in the inner loop
          
          for y in nh:  # Inner loop to search through the neighborhood
              self.parameters['x'] = y.copy()
              logging.info(f"Test schedule = {self.parameters['x']}")
              self.state = 0
              self._initialize_system()
              self.calculate_system_states(until=len(self.parameters['x']))
              logging.info("System recalculated")
              self.calculate_wait_times()
              test_wt = np.sum(self.system["ew"])  # Use np.sum instead of list's sum method
              logging.info(f"Average waiting time={test_wt / np.sum(self.parameters['x'])}")
              self.calculate_tardiness()
              indices = np.arange(self.system['p_min'][-1].size) 
              exp_tard = (indices * self.system['p_min'][-1]).sum()
              logging.info(f'Expected tardiness={exp_tard}')
              test_loss = omega * test_wt / np.sum(self.parameters['x']) + (1 - omega) * exp_tard
              logging.info(f'obj_value = {test_loss}')
              search_path.append({'schedule': self.parameters['x'].copy(), 'loss': test_loss})  # Save current schedule and loss
              if test_loss < lowest_loss:
                  lowest_loss = test_loss
                  store_optim['x'] = self.parameters['x'].copy()
                  newsystem = copy.deepcopy(self.system)
                  store_optim['system'] = newsystem
                  store_optim['tot_wt'] = test_wt
                  logging.info(f'{datetime.datetime.now()} - Found lower loss = {lowest_loss} with x = {store_optim["x"]} and system = {store_optim["system"]}')
                  improved = True
                  break  # Exit the inner loop to generate a new neighborhood
          
          if not improved:  # If no improvement was found in the inner loop
              logging.info(f'Finished searching')
              self.parameters['x'] = store_optim['x'].copy()
              self.system['p_min'] = store_optim['system']['p_min']
              self.system['p_plus'] = store_optim['system']['p_plus']
              self.system['w'] = store_optim['system']['w']
              self.system['ew'] = store_optim['system']['ew']
              self.system['search_path'] = search_path
              break  # Exit the outer loop - the search is complete
      
  def __str__(self):
      return "p_min = % s \nw = % s \np_plus = % s \new = % s \ntardiness = % s \nloss = % s" % (self.system['p_min'], self.system['w'], self.system['p_plus'], self.system['ew'], self.system['p_min'][-1], self.system['loss'])

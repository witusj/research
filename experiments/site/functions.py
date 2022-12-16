## Functions
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

### Lognormal mean and lns / see: https://en.wikipedia.org/wiki/Log-normal_distribution

def logn_mean_lns(lnm, lns):
  sigma = np.sqrt(np.log(1 + (lns/lnm)**2))
  mu = np.log(lnm**2 / np.sqrt(lnm**2 + lns**2))
  return np.array([mu, sigma])

### Random lognormal service times generator
def generate_logn_sts(s, n, lnm, lns):
  sts = np.random.lognormal(mean = logn_mean_lns(lnm, lns)[0], sigma=logn_mean_lns(lnm, lns)[1], size=s * n)
  return np.reshape(sts, (s, n))

##############
#### TEST ####
generate_logn_sts(5, 5, 14, 10)
generate_logn_sts(1, 8, 14, 10)
##############

### Client generator
def generate_client(cts, sts, pct, ns=0):
  ct = np.random.choice(cts, size = 1, p=pct)[0] # generate random client type
  st = np.random.binomial(n=1, p=1-ns)*np.random.choice(sts, size = 1)[0] # generate random service time / if no-show -> st = 0
  return np.array([ct, st])

### Simulation
### ATTENTION: handling of emergency patients has not yet been implemented

def simulate(iats, cts, sts, pct, ns, logs=False):
  wt = 0 # initial value waiting time
  wts = [None] * len(iats) # array for saving waiting times
  for i in range(len(iats)):
    if(i != 0): # don't calculate waiting time for first client in schedule
      wt = max(0, tis - iats[i]) # calculate waiting time
    ct, st = generate_client(cts, sts, pct, ns) # client type and service time
    tis = wt + st # calculate time in system
    wts[i] = wt
    if(logs): print(iats[i], wt, st, tis, ct)
  return np.array(wts)

def simulate_crn(iats, cts, sts, logs=False):
  wt = 0 # initial value waiting time
  wts = [None] * len(iats) # array for saving waiting times 
  for i in range(len(iats)):
    if(i != 0): # don't calculate waiting time for first client in schedule
      wt = max(0, tis - iats[i]) # calculate waiting time
    ct =  cts[i] # client type
    st =  sts[i] # service time
    tis = wt + st # calculate time in system
    wts[i] = wt
    if(logs): print(iats[i], wt, st, tis, ct)
  return np.array(wts), tis

def transform_iats_schedule(iats: list, d: int, T: int):
  iats = np.array(iats)
  ats = np.cumsum(iats)
  sats = np.arange(d*(T+1),step = d)
  schedule = np.histogram(ats, bins=sats)
  return schedule

##############
#### TEST ####
transform_iats_schedule([0, 0 ,30, 0, 60, 0, 0, 0, 60, 0, 0, 0], d = 15, T = 11)
##############

def transform_schedule_iats(schedule: list, d: int):
  schedule = np.array(schedule)
  T = schedule.size
  sats = np.arange(d*T,step = d)
  ats = np.repeat(sats, schedule)
  iats = np.diff(ats)
  iats = np.insert(iats, 0, ats[0])
  iats = iats.astype(int)
  return iats

##############
#### TEST ####
schedule = transform_iats_schedule([0, 0 ,30, 0, 60, 0, 0, 0, 60, 0, 0, 0], d = 15, T = 11)[0]
transform_schedule_iats(schedule, d = 15)
##############

def patient_shift_matrix(schedule):
  T = len(schedule) # number of intervals
  neg = -np.eye(T) # create matrix with diagonal negative ones
  pos = np.eye(T, k=-1) # create matrix with diagonal ones shifted to left
  psm = neg + pos # combine two matrices
  psm[0, T-1] = 1 # add a one to the end of the first row
  sschedules = psm + schedule # create matrix of schedules with one patient shifting from k to k-1
  sschedules = np.insert(sschedules, 0, schedule, axis=0) # add original schedule

  return sschedules[sschedules.min(axis=1)>=0, :] # only return schedules that have non-negative arrivals

##############
#### TEST ####
schedule = np.zeros(12)
i = np.arange(12, step=2)
schedule[i] = 1
schedule[-1] = 1
schedule[0] = 2
patient_shift_matrix(schedule)
##############

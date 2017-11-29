import numpy as np

def  simulate_prizedoor(nsim):
    return np.random.randint(0,3,(nsim))

def simulate_guess(nsim):
    return np.zeros(nsim, int)

def goat_door(prizedoors,guesses):
    result = np.random.randint(0,3,prizedoors.size)
    while True:
        bad = (result == prizedoors) | (result == guesses)
        if not bad.any(axis=0):
            return result
        result[bad] = np.random.randint(0,3,bad.sum())

def switch_guess(guesses, goatdoors):
    result = np.zeros(guesses.size)
    switch = {(0, 1): 2, (0, 2): 1, (1, 0): 2, (1, 2): 1, (2, 0): 1, (2, 1): 0}
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            mask = (guesses == i) & (goatdoors == j)
            if not mask.any():
                continue
            result = np.where(mask, np.ones_like(result) * switch[(i, j)], result)
    return result
#print switch_guess(np.array([1,2,1]),np.array([2,1,0]))

def win_percentage(guesses,prizedoors):
    return 100*(guesses == prizedoors).mean()
#print win_percentage(np.array([1,2,1]),np.array([1,1,0]))

guess = simulate_guess(10000)
prize = simulate_prizedoor(10000)
goats = goat_door(prize,guess)
switch = switch_guess(guess,goats)
print win_percentage(guess,prize)
print win_percentage(switch,prize)


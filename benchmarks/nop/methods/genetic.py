import sys
import time
import math
import random


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# returns a path (list of points) through s with high value
def ellinit_replacement(s1, start_point, end_point, tmax):
    s = list(s1)
    path = [start_point, end_point]
    length = distance(start_point, end_point)
    found = True
    while found and len(s) > 0:
        min_added_length = -1
        max_added_reward = 0
        for j in range(len(s)):
            for k in range(len(path) - 1):
                added_length = (distance(path[k], s[j]) +
                                distance(path[k + 1], s[j]) -
                                distance(path[k], path[k + 1]))  # optimize later
                if length + added_length < tmax and s[j][2] > max_added_reward:
                    min_added_length = added_length
                    max_added_reward = s[j][2]
                    minpoint = j
                    pathpoint = k + 1
        if min_added_length > 0:
            # add to path
            path.insert(pathpoint, s.pop(minpoint))
            length = length + min_added_length
        else:
            found = False
    return path


# returns a list of L paths with the best path in the first position
# by weight rather than length
def init_replacement(s1, start_point, end_point, tmax):
    s = list(s1)
    L = len(s) if len(s) <= 10 else 10
    if L == 0:
        # print 'something is probably wrong'
        # actually maybe not
        return [[start_point, end_point]]

    # decorate and sort by weight
    dsub = sorted([(x[4], x) for x in s])[::-1]  # this is different
    ls = dsub[:L]
    rest = dsub[L:]
    paths = []
    for i in range(L):
        path = [start_point, ls[i][1], end_point]
        length = distance(path[0], path[1]) + distance(path[1], path[2])
        assert (length < tmax)
        arest = ls[:i] + ls[i + 1:] + rest
        arest = [x[1] for x in arest]  # undecorate
        assert (len(arest) + len(path) == len(s) + 2)
        found = True
        while found == True and len(arest) > 0:
            min_added_length = -1
            max_weight = 0
            for j in range(len(arest)):
                for k in range(len(path) - 1):
                    added_length = (distance(path[k], arest[j]) +
                                    distance(path[k + 1], arest[j]) -
                                    distance(path[k], path[k + 1]))  # optimize later
                    if length + added_length < tmax and arest[j][4] < max_weight:
                        min_added_length = added_length
                        max_weight = arest[j][4]
                        minpoint = j
                        pathpoint = k + 1
            if min_added_length > 0:
                # add to path
                path.insert(pathpoint, arest.pop(minpoint))
                length = length + min_added_length
            else:
                found = False
        if length < tmax:
            paths.append(path)

    assert (len(paths) > 0)
    return [x[1] for x in sorted([(sum([y[2] for y in z]), z) for z in paths])[::-1]]


# returns the subset of s that is on/in the ellipse defined by foci f1, f2 and the major axis
def ell_sub(axis, f1, f2, s):
    result = []
    for item in s:
        if distance(item, f1) + distance(item, f2) <= axis:
            result.append(item)
    return result


# returns a list of L paths with the best path in the first position
def initialize(s, start_point, end_point, tmax):
    L = len(s) if len(s) <= 10 else 10
    if L == 0:
        return [[start_point, end_point]]

    dsub = sorted([(distance(x, start_point) + distance(x, end_point), x) for x in s]
                  )[::-1]  # optimize later
    ls = dsub[:L]
    rest = dsub[L:]
    paths = []
    for i in range(L):
        path = [start_point, ls[i][1], end_point]
        length = ls[i][0]
        assert (length == distance(path[0], path[1]) + distance(path[1], path[2]))
        arest = ls[:i] + ls[i + 1:] + rest
        arest = [x[1] for x in arest]  # undecorate
        assert (len(arest) + len(path) == len(s) + 2)
        found = True
        while found == True and len(arest) > 0:
            min_added = -1
            for j in range(len(arest)):
                for k in range(len(path) - 1):
                    added_length = (distance(path[k], arest[j]) +
                                    distance(path[k + 1], arest[j]) -
                                    distance(path[k], path[k + 1]))  # optimize later
                    if length + added_length < tmax and (added_length < min_added or min_added < 0):
                        min_added = added_length
                        minpoint = j
                        pathpoint = k + 1
            if min_added > 0:
                # add to path
                path.insert(pathpoint, arest.pop(minpoint))
                length = length + min_added
            else:
                found = False
        paths.append(path)

    assert (len([x[1] for x in sorted([(sum([y[2] for y in z]), z) for z in paths]
                                      )[::-1]]) > 0)
    return [x[1] for x in sorted([(sum([y[2] for y in z]), z) for z in paths])[::-1]]


# fitness will take a set s and a set of weights and return a tuple containing the fitness and the best path
def fitness(chrom, s, start_point, end_point, tmax):
    augs = []
    for i in range(len(s)):
        augs.append((s[i][0],
                     s[i][1],
                     s[i][2],
                     s[i][3],
                     s[i][4] + chrom[i]))
    if debug:
        print('fitness---------------------------------')
        print('augs:')
        print(augs)
    # best = oph.ellinit_replacement( augs, start_point, end_point, tmax )
    ellset = ell_sub(tmax, start_point, end_point, augs)
    # best = oph.initialize( ellset, start_point, end_point, tmax )[0]
    best = init_replacement(ellset, start_point, end_point, tmax)[0]
    if debug:
        print('best:')
        print(best)
        print('best real reward:')
        print([x[3] for x in best])
        print(len(s))
        print([s[x[3] - 2] for x in best[1:len(best) - 1]])
        print([s[x[3] - 2][2] for x in best[1:len(best) - 1]])
        print((sum([s[x[3] - 2][2] for x in best[1:len(best) - 1]]), best))
    return sum([s[x[3] - 2][2] for x in best[1:len(best) - 1]]), best


def crossover(c1, c2):
    assert (len(c1) == len(c2))
    point = random.randrange(len(c1))
    first = random.randrange(2)
    if (first):
        return c1[:point] + c2[point:]
    else:
        return c2[:point] + c1[point:]


def mutate(chrom, mchance, msigma):
    return [x + random.gauss(0, msigma) if random.randrange(mchance) == 0 else
            x for x in chrom]


def run_alg_f(f, tmax, N):
    random.seed()
    cpoints = []
    _ = f.readline()  # ignore first line of file
    for i in range(N):
        cpoints.append(tuple([float(x) for x in f.readline().split()]))
    if debug:
        print('N:            ', N)
    return solve_op_genetic(cpoints, tmax)


def solve_op_genetic(points, tmax, return_sol=False, verbose=True):
    cpoints = [tuple(p) + (i, 0) for i, p in enumerate(points)]
    start_point = cpoints.pop(0)
    end_point = cpoints.pop(0)
    assert distance(start_point, end_point) < tmax
    popsize = 10
    genlimit = 10
    kt = 5
    isigma = 10
    msigma = 7
    mchance = 2
    elitismn = 2
    if debug:
        print('data set size:', len(cpoints) + 2)
        print('tmax:         ', tmax)
        print('parameters:')
        print('generations:     ', genlimit)
        print('population size: ', popsize)
        print('ktournament size:', kt)
        print('mutation chance: ', mchance)
        print(str(elitismn) + '-elitism')

    start_time = time.process_time()
    # generate initial random population
    pop = []
    for i in range(popsize + elitismn):
        chrom = []
        for j in range(len(cpoints)):
            chrom.append(random.gauss(0, isigma))
        chrom = (fitness(chrom, cpoints, start_point, end_point, tmax)[0], chrom)
        while i - j > 0 and j < elitismn and chrom > pop[i - 1 - j]:
            j += 1
        pop.insert(i - j, chrom)

    bestfit = 0
    for i in range(genlimit):
        nextgen = []
        for j in range(popsize):
            # select parents in k tournaments
            parents = sorted(random.sample(pop, kt))[kt - 2:]  # optimize later
            # crossover and mutate
            offspring = mutate(crossover(parents[0][1], parents[1][1]), mchance, msigma)
            offspring = (fitness(offspring, cpoints, start_point, end_point, tmax)[0], offspring)
            if offspring[0] > bestfit:
                bestfit = offspring[0]
                if verbose:
                    print(bestfit)
            if elitismn > 0 and offspring > pop[popsize]:
                l = 0
                while l < elitismn and offspring > pop[popsize + l]:
                    l += 1
                pop.insert(popsize + l, offspring)
                nextgen.append(pop.pop(popsize))
            else:
                nextgen.append(offspring)
        pop = nextgen + pop[popsize:]

    bestchrom = sorted(pop)[popsize + elitismn - 1]
    end_time = time.process_time()

    if verbose:
        print('time:')
        print(end_time - start_time)
        print('best fitness:')
        print(bestchrom[0])
        print('best path:')
    best_path = fitness(bestchrom[1], cpoints, start_point, end_point, tmax)[1]
    if verbose:
        print([x[3] for x in best_path])

        print('their stuff:')
    stuff = initialize(ell_sub(tmax, start_point, end_point, cpoints), start_point, end_point, tmax)[0]
    if verbose:
        print('fitness:', sum([x[2] for x in stuff]))
        print('my stuff:')
    stuff2 = ellinit_replacement(cpoints, start_point, end_point, tmax)
    if verbose:
        print('fitness:', sum([x[2] for x in stuff2]))
        print('checking correctness...')
    total_distance = (distance(start_point, cpoints[best_path[1][3] - 2]) +
                      distance(end_point, cpoints[best_path[len(best_path) - 2][3] - 2]))
    for i in range(1, len(best_path) - 3):
        total_distance += distance(cpoints[best_path[i][3] - 2], cpoints[best_path[i + 1][3] - 2])
    if verbose:
        print('OK' if total_distance <= tmax else 'not OK')
        print('tmax:          ', tmax)
        print('total distance:', total_distance)
    if return_sol:
        return bestchrom[0], best_path, end_time - start_time
    return bestchrom[0], end_time - start_time


if __name__ == '__main__':
    debug = True if 'd' in sys.argv else False
    solve_op_genetic(open(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
else:
    debug = False

import numpy as np
import networkx as nx
import json
from utils import SIR_net_adaptive, initNET_rnd
import sys
import multiprocessing as mp
import time

# import parameters of the simulation
with open('parameters.txt') as f:
    raw_par = f.read()
par = json.loads(raw_par)

nsim = int(sys.argv[1]) # number of simulations

# output file
# time pol r attak_rate clustering_mean clustering ave/std stat/dyn

filename = 'Simulations/SIR_simulation_novax.csv'
output = open(filename, 'a')
columns = ['time,I,nv,c1,c2,ar,V_tot,kind,net_type']
output.writelines(columns)
output.write('\n')

r = 0.1   # Vaccination rate
pol = 0.3 # Polarization

# random generator
rng = np.random.default_rng(2022)

def simulation_step(par, rng, r, pol, nv_init):
    seed = rng.integers(1, 10000)
    initial_novax = rng.choice(np.arange(par['N']), nv_init)
    initial_infecteds = rng.choice(np.arange(par['N']), par['n_infecteds'])
    phys_net = nx.barabasi_albert_graph(par['N'], int(par['ave_degree']/2))
    info_net_stat = phys_net.copy()
    initNET_rnd(info_net_stat, initial_novax=initial_novax)
    info_net_dyn = info_net_stat.copy()

    time_stat, _, I_stat, _, V_stat, I_tot_stat, c1_stat, c2_stat = SIR_net_adaptive(
        phys_net, info_net_stat,
        beta=par['beta'],
        mu=par['mu'],
        r=r,
        pro=par['pro'],
        pol=pol,
        initial_infecteds=initial_infecteds,
        rewiring=False,
        rng=np.random.default_rng(seed),
        message=False)

    time_dyn, _, I_dyn, _, V_dyn, I_tot_dyn, c1_dyn, c2_dyn = SIR_net_adaptive(
        phys_net, info_net_dyn,
        beta=par['beta'],
        mu=par['mu'],
        r=r,
        pro=par['pro'],
        pol=pol,
        initial_infecteds=initial_infecteds,
        rewiring=True,
        rng=np.random.default_rng(seed),
        message=False)

    return(
        [len(time_stat), len(time_dyn)],
        [I_stat, I_dyn],
        [I_tot_stat, I_tot_dyn],
        [c1_stat, c1_dyn],
        [c2_stat, c2_dyn],
        [V_stat[-1], V_dyn[-1]])

def simulate_params(r, pol, par, nv_init, rng, out_file):
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(simulation_step, args=(par, rng, r, pol, nv_init)) for _ in range(nsim)]
    answers = [res.get(timeout=600) for res in results]
    pool.close()

    answers = np.array(answers, dtype=object) # [nsim, 6, 2]
    static_data = answers[:, :, 0] # [nsim, 6]
    dynamic_data = answers[:, :, 1]# [nsim, 6]
    min_time_stat = min(static_data[:, 0])
    min_time_dyn = min(dynamic_data[:, 0])

    mean_I_stat = np.mean([static_data[i, 1][:min_time_stat] for i in range(nsim)], axis=0)
    std_I_stat = np.std([static_data[i, 1][:min_time_stat] for i in range(nsim)], axis=0)
    mean_I_dyn = np.mean([dynamic_data[i, 1][:min_time_dyn] for i in range(nsim)], axis=0)
    std_I_dyn = np.std([dynamic_data[i, 1][:min_time_dyn] for i in range(nsim)], axis=0)

    mean_I_tot_stat = np.mean(static_data[:, 2])
    std_I_tot_stat = np.std(static_data[:, 2])
    mean_I_tot_dyn = np.mean(dynamic_data[:, 2])
    std_I_tot_dyn = np.std(dynamic_data[:, 2])

    mean_c1_stat = np.mean([static_data[i, 3][:min_time_stat] for i in range(nsim)], axis=0)
    std_c1_stat = np.std([static_data[i, 3][:min_time_stat] for i in range(nsim)], axis=0)
    mean_c1_dyn = np.mean([dynamic_data[i, 3][:min_time_dyn] for i in range(nsim)], axis=0)
    std_c1_dyn = np.std([dynamic_data[i, 3][:min_time_dyn] for i in range(nsim)], axis=0)

    mean_c2_stat = np.mean([static_data[i, 4][:min_time_stat] for i in range(nsim)], axis=0)
    std_c2_stat = np.std([static_data[i, 4][:min_time_stat] for i in range(nsim)], axis=0)
    mean_c2_dyn = np.mean([dynamic_data[i, 4][:min_time_dyn] for i in range(nsim)], axis=0)
    std_c2_dyn = np.std([dynamic_data[i, 4][:min_time_dyn] for i in range(nsim)], axis=0)

    mean_V_tot_stat = np.mean(static_data[:, 5])
    std_V_tot_stat = np.std(static_data[:, 5])
    mean_V_tot_dyn = np.mean(dynamic_data[:, 5])
    std_V_tot_dyn = np.std(dynamic_data[:, 5])

    times_stat = np.arange(min_time_stat)
    times_dyn = np.arange(min_time_dyn)

    for t, i_mean, i_std, c1_mean, c2_mean, c1_std, c2_std in zip(times_stat, mean_I_stat, std_I_stat, mean_c1_stat, mean_c2_stat, std_c1_stat, std_c2_stat):
        out_file.write(f'{t},{round(i_mean, 2)},{nv_init},{round(c1_mean, 3)},{round(c2_mean, 3)},{int(mean_I_tot_stat)},{round(mean_V_tot_stat,3)},mean,static')
        out_file.write('\n')
        out_file.write(f'{t},{round(i_std, 2)},{nv_init},{round(c1_std, 3)},{round(c2_std, 3)},{int(std_I_tot_stat)},{round(std_V_tot_stat,3)},std,static')
        out_file.write('\n')

    for t, i_mean, i_std, c1_mean, c2_mean, c1_std, c2_std in zip(times_dyn, mean_I_dyn, std_I_dyn, mean_c1_dyn, mean_c2_dyn, std_c1_dyn, std_c2_dyn):
        out_file.write(f'{t},{round(i_mean, 2)},{nv_init},{round(c1_mean, 3)},{round(c2_mean, 3)},{int(mean_I_tot_dyn)},{round(mean_V_tot_dyn,3)},mean,dynamic')
        out_file.write('\n')
        out_file.write(f'{t},{round(i_std, 2)},{nv_init},{round(c1_std, 3)},{round(c2_std, 3)},{int(std_I_tot_dyn)},{round(std_V_tot_dyn,3)},std,dynamic')
        out_file.write('\n')
    print(f'completed: nv_init={nv_init}')

start_tot = time.time()
for nv_init in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
    start = time.time()
    simulate_params(r, pol, par, nv_init, rng, output)
    stop = time.time()
    print('simulation took:', round((stop - start)/60, 1), 'min')
    print('total time:', round((stop - start_tot)/60, 1), 'min', '\n')
stop_tot = time.time()
output.close()
print('\nSimulation completed')
print('total simulation time:', round((stop_tot - start_tot)/60, 1), 'min')
import numpy as np   
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy
from scipy import optimize
import sys

# Simple SIR process
def SIR_net(G, beta, mu, initial_infecteds, seed=123):
    """G = network, beta = infection rate, mu = recovery rate,
    initial_infecteds = list of infected nodes at time t=0 """

    np.random.seed(seed)
    random.seed(seed)

    #INITIALIZATION
    inf_status = {}  # infectious status of a node: 0 = susceptible; 1 = infectious; 2 = recovered 
    for i in range(G.number_of_nodes()):
        inf_status[i] = 0
    for i in initial_infecteds:
        inf_status[i] =1  
    nx.set_node_attributes(G, inf_status, 'inf_status')
    nx.set_node_attributes(G, inf_status, 'new_inf_status')

    #SIMULATION
    # one single iteration
    N = G.number_of_nodes()
    time = []
    S = []
    I = []
    R = []
    time.append(0)
    S.append(N - len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    t=0
    while True:
        t+=1
        time.append(t)
        # transmission and recovery
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 1:
                if random.random() < mu:
                    G.nodes[i]['new_inf_status'] = 2
            elif G.nodes[i]['inf_status'] == 0:
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status'] == 1:
                        if random.random() < beta:
                            G.nodes[i]['new_inf_status'] = 1
                            break  

        # update infectious status                    
        for i in nx.nodes(G):
            G.nodes[i]['inf_status'] = G.nodes[i]['new_inf_status']

        # compute total number of susceptible, infectious, recovered
        suscep = 0
        infect = 0
        recov = 0
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 0:
                suscep += 1
            elif G.nodes[i]['inf_status'] == 1:
                infect += 1
            elif G.nodes[i]['inf_status'] == 2:
                recov += 1
        S.append(suscep)
        I.append(infect)
        R.append(recov)

        # end the simulation if no mode infectious
        print(f'simulation until time t={t+1}', end='\r')
        sys.stdout.flush()
        if infect == 0:
            break

    return np.array(time), np.array(S), np.array(I), np.array(R)

def initNET_SI(G, I0, I_seed):
    """Initialize the information network with a SI model.
    G = network,
    I0 = number of desired initial infecteds,
    I_seed = list of infecteds to start the initialization with"""

    aware_status = {}
    for i in range(G.number_of_nodes()):
        aware_status[i] = 0
    for i in I_seed:
        aware_status[i] = 1
    nx.set_node_attributes(G, aware_status, 'aware_status')
    nx.set_node_attributes(G, aware_status, 'new_aware_status')
    actual_infected = len(I_seed)
    # SI dynamics
    while actual_infected < I0:
        for i in G.nodes():
            if G.nodes[i]['aware_status'] == 0:                                                
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['aware_status'] == 1:
                        if random.random() < 0.1:                                             
                            if G.nodes[i]['new_aware_status'] != 1 and actual_infected < I0:
                                G.nodes[i]['new_aware_status'] = 1
                                actual_infected += 1
                            
        for i in nx.nodes(G):
                G.nodes[i]['aware_status'] = G.nodes[i]['new_aware_status']

def initNET_rnd(G, initial_novax):
    """Initialize the information network randomly from the initial_novax list"""
    aware_status = {}
    for i in range(G.number_of_nodes()):
        aware_status[i] = 0
    for i in initial_novax:
        aware_status[i] = 1
    nx.set_node_attributes(G, aware_status, 'aware_status')
    nx.set_node_attributes(G, aware_status, 'new_aware_status')


# We use just 2 states in the information network: PV and NV. The interaction between the two
# follows a voter model, plus there is the effet of classical media acting on the NV population (a small effect that 
# should account for the fact that, as time goes by, social and political pressure erode the NV population)

def SIR_net_adaptive(G, NET, beta, mu, r, pro, pol, initial_infecteds, rewiring=True, rng=np.random.default_rng(123), message=True):
    """
    G: physical network
    NET: information network,
    beta: infection rate,
    mu: recovery rate,
    r: rate of vaccination for PV,
    pro: rate of classical media influence on people,
    pol: propensity of opinion polarization,
    initial_infecteds: list of infected nodes at time t=0,
    rewiring: whether the information network should be static or dynamic"""
    
    #INITIALIZATION
    inf_status = {}
    got_infected = {}
    for i in range(G.number_of_nodes()):
        inf_status[i] = 'S'
        got_infected[i] = 0
    for i in initial_infecteds:
        inf_status[i] = 'I'
        got_infected[i] = 1
    nx.set_node_attributes(G, inf_status, 'inf_status')
    nx.set_node_attributes(G, inf_status, 'new_inf_status')
    nx.set_node_attributes(G, got_infected, 'got_infected')
    
    #SIMULATION
    # one single iteration
    N = G.number_of_nodes()
    time = []
    S = []
    I = []
    R = []
    V = []
    NV = [] # No-Vax
    PV = [] # Pro-Vax
    
    time.append(0)
    
    # SIR
    S.append(N - len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    V.append(0)
    
    # Information network initialized outside the function
    # 0 = pro vax, 1 = no vax
    nv_init = 0
    pv_init = 0
    for i in NET.nodes():
        if NET.nodes[i]['aware_status'] == 0:
            pv_init += 1
        else:
            nv_init += 1
    NV.append(nv_init)
    PV.append(pv_init)
    
    
    ###################################
    # How many people did change idea?
    lista = []
    a = np.zeros(N)
    lista.append(a)

    changers = []
    bichangers = []
    
    changers.append(0)
    bichangers.append(0)
    
    ###################################
    
    t = 0
    total_infected = 0
    while True:
        

        lista.append(a)
        t += 1
        time.append(t)
        
        # REWIRING OF THE INFORMATION NETWORK 
        if rewiring:                           
            novax   = []
            provax  = []
        
            for u in nx.nodes(NET):
                if NET.nodes[u]['aware_status'] == 0:
                    provax.append(u)
                else:
                    novax.append(u)  

            for i in nx.edges(NET):      
                i = list(i)  
                if (NET.nodes[i[0]]['aware_status'] + NET.nodes[i[1]]['aware_status'] == 1):      # two nodes with different opinions 
                    if rng.random() < pol:    
                        NET.remove_edge(i[0], i[1])
                        if NET.nodes[i[0]]['aware_status'] == 1:
                            NET.add_edge(i[0], rng.choice(novax)) 
                            NET.add_edge(i[1], rng.choice(provax))
                        else:
                            NET.add_edge(i[1], rng.choice(novax)) 
                            NET.add_edge(i[0], rng.choice(provax))

        # EPIDEMICS IN THE PHYSICAL NETWORK
        
        ###################################
        bichanger = 0
        changer = 0
        ###################################
        
        
        for i in nx.nodes(G):
            # all possible transitions
            if (NET.nodes[i]['aware_status'] == 0) and (G.nodes[i]['inf_status'] == 'S'):        # provax that get vaccinated
                if rng.random() < r:
                    G.nodes[i]['new_inf_status'] = 'V'
                    
            if G.nodes[i]['inf_status'] == 'I':                                                  # infectious that recover
                if rng.random() < mu:
                    G.nodes[i]['new_inf_status'] = 'R'
                    
            elif G.nodes[i]['inf_status'] == 'S':                                                # susceptible
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status'] == 'I':
                        if rng.random() < beta:                                                  # here they get the disease
                            G.nodes[i]['new_inf_status'] = 'I'
                            G.nodes[i]['got_infected'] = 1
                            
            # EPIDEMICS IN THE INFORMATION NETWORK
            
            if rng.random() < pro:
                NET.nodes[i]['new_aware_status'] = 0                                         # become a pro vax due to classical media
                
            else:                                                                            # or look around into social media
                if G.nodes[i]['inf_status'] != 'V':
                    neighbors = list(nx.all_neighbors(NET, i))
                    target = rng.choice(neighbors, 1)[0]
                    NET.nodes[i]['new_aware_status'] = NET.nodes[target]['aware_status']     # can become a pro/no vax via neighbours
                    
                    ###############################################################################
                    if NET.nodes[target]['aware_status'] != NET.nodes[target]['new_aware_status']:
                        lista[t][target] = 1
                    if lista[t-1][target] == 1 :
                        changer = changer +1
                    if lista[t-2][target] == 1 and lista[t-1][target] == 1:
                        bichanger = bichanger +1
        #print(changer)
            
        changers.append(changer)
        bichangers.append(bichanger)
        
        ############################################################################################
        
        # UPDATE NETWORKS     
        for i in nx.nodes(G):
            G.nodes[i]['inf_status'] = G.nodes[i]['new_inf_status']
        for i in nx.nodes(NET):
            NET.nodes[i]['aware_status'] = NET.nodes[i]['new_aware_status']


        # COMPUTE THE TOTAL NUMBER OF SUSCEPTIBLE, INFECTED AND RECOVERED PEOPLE
        suscep = 0
        infect = 0
        recov = 0
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 'S':
                suscep += 1
            elif G.nodes[i]['inf_status'] == 'I':
                infect += 1
            elif G.nodes[i]['inf_status'] == 'R':
                recov += 1
        vaccin = len(nx.nodes(G)) - suscep - infect - recov
        S.append(suscep)
        I.append(infect)
        R.append(recov)
        V.append(vaccin)

        # end simulation if no more infectious are present
        if message:
            print(f'simulation until time t={t+1}', end='\r')
            sys.stdout.flush() 
        if infect == 0:
            for _, data in G.nodes(data=True):
                total_infected += data['got_infected']
            break


    return np.array(time), np.array(S), np.array(I), np.array(R), np.array(V), total_infected, changers, bichangers

def plot_info_network(G):
    nv_list = []
    pv_list = []
    for i in G.nodes():
        if G.nodes[i]['aware_status'] == 0:
            pv_list.append(i)
        else:
            nv_list.append(i)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_nodes(G, pos=pos, nodelist=pv_list,
        node_color='blue', label='Pro-Vax', node_size=20)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nv_list,
        node_color='red', label='No-Vax', node_size=20)
    nx.draw_networkx_edges(G, pos=pos, width=0.1)
    plt.legend(scatterpoints=1, fontsize=12)
    
    
    
    
    
def fitR0(x, y, start = 0, mu = 1/7, n_points = 6, log = True, info=True):
    xx = x
    yy = y
    xfit = xx[start:start+n_points]
    yfit = yy[start:start+n_points]
    
    def exp(X, I0, G):
        return I0*np.exp(G*X)
    
    # fit curve
    popt, _ = scipy.optimize.curve_fit(exp, xfit, yfit)
    
    I0, G = popt
    
    if info:
        # create a line plot for the mapping function
        x_line = np.linspace(xfit[0], xfit[-1], 20)
        y_line = exp(x_line, I0, G)
        plt.scatter(xx, yy)
        plt.plot(x_line, y_line, '--', color='red')
        if log:
            plt.yscale("log")
        plt.show()
    
        print("G =", G)
        print("\nR0 =", G/mu+1)
    return G/mu+1
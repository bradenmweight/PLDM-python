import numpy as np
import multiprocessing as mp
import time
import model

dtE = model.parameters.dtE
dtI = model.parameters.dtI
NSteps = model.parameters.NSteps
NTraj = model.parameters.NTraj
NStates = model.parameters.NStates
M = model.parameters.M

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "focused"):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    qF = np.zeros((Nstates))
    qB = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    pB = np.zeros((Nstates))
    if (stype == "focused"):
        qF[initState] = 1.0
        qB[initState] = 1.0
        pF[initState] = 1.0
        pB[initState] = -1.0 # This minus sign allows for backward motion of fictitious oscillator
    else:
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

def propagateMapVars(qF, qB, pF, pB, dt, R):
    VMat = model.Hel(R)
    qFin, qBin, pFin, pBin = qF, qB, pF, pB # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step
    VMatxqB =  np.array([np.sum(VMat[k,:] * qBin[:]) for k in range(NStates)])
    VMatxqF =  np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])
    for i in range(NStates): # Loop over q's and p's for initial update of positions
       # Update momenta using input positions (first-order in dt)
       pB[i] -= 0.5 * dt * np.sum(VMat[i,:] * qBin[:]) ## First Derivatives ##
       pF[i] -= 0.5 * dt * np.sum(VMat[i,:] * qFin[:])
       # Now update positions with input momenta (first-order in dt)
       qB[i] += dt * np.sum(VMat[i,:] * pBin[:])
       qF[i] += dt * np.sum(VMat[i,:] * pFin[:])
       for k in range(NStates):
           # Update positions to second order in dt
           qB[i] -= (dt**2/2.0) * (VMat[i,k])* VMatxqB[k] ## Second Derivatives ##
           qF[i] -= (dt**2/2.0) * (VMat[i,k])* VMatxqF[k]
    
    # Update momenta using output positions (first-order in dt)
    for i in range(NStates): # Loop over q's and p's for final update of fictitious momentum
       pB[i] -= 0.5 * dt * np.sum(VMat[i,:] * qB[:])
       pF[i] -= 0.5 * dt * np.sum(VMat[i,:] * qF[:])
    return qF, qB, pF, pB

def Force(R, qF, qB, pF, pB):
    dHel = model.dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates
    F = np.zeros((len(R)))
    for i in range(len(qF)):
        for j in range(len(qF)):
            F -= 0.25 * dHel[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
    return F

def VelVerF(R, P, qF, qB, pF, pB, dtI, dtE=dtI/20, M=1): # Ionic position, ionic velocity, etc.
    v = P/M
    F1 = Force(R, qF, qB, pF, pB)
    R += v * dtI + 0.5 * F1 * dtI ** 2 / M
    EStep = int(dtI/dtE)
    for t in range(EStep):
        qF, qB, pF, pB = propagateMapVars(qF, qB, pF, pB, dtE, R)
    F2 = Force(R, qF, qB, pF, pB)
    v += 0.5 * (F1 + F2) * dtI / M
    return R, v*M, qF, qB, pF, pB

def getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0):
    #print (step/NSteps * 100, "%")
    rho = np.zeros(( len(qF), len(qF) ), dtype=complex) # Define density matrix
    rho0 = (qF0 - 1j*pF0) * (qB0 + 1j*pB0)
    for i in range(len(qF)):
       for j in range(len(qF)):
          rho[i,j] = 0.25 * (qF[i] + 1j*pF[i]) * (qB[j] - 1j*pB[j]) * rho0
    return rho

def RunIterations(n):
    #print (n)
    #for itraj in range(int(NTraj/Ncpus)):
    #print (itraj, NTraj/Ncpus)
    #print (rho_ensemble[0,0,0].real)
    rho_dum = np.zeros((NStates,NStates,NSteps), dtype=complex)
    R,P = model.initR()
    qF, qB, pF, pB = initMapping(NStates,initState) # Call function to initialize fictitious oscillators according to focused ("Default") or according to gaussian random distribution
    qF0, qB0, pF0, pB0 = qF[initState], qB[initState], pF[initState], pB[initState] # Set initial values of fictitious oscillator variables for future use
    for i in range(NSteps): # One trajectory
        if (i % 1 == 0):
            rho_current = getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0)
            rho_dum[:,:,i] = rho_current
        R, P, qF, qB, pF, pB = VelVerF(R, P, qF, qB, pF, pB, dtI, dtE, M)
    return rho_dum


## Start Main Program

initState = 0 # Choose (arbitrarily???) the initial state of the particle population.

Ncpus = 24
#Ncpus = mp.cpu_count() # Gives wrong number of CPUs
rho_ensemble = np.zeros((NStates,NStates,NSteps), dtype=complex)
print (f"There will be {Ncpus} cores with {NTraj} trajectories.")

start = time.time()

runList = np.arange(NTraj)
with mp.Pool(processes=Ncpus) as pool:
    result = pool.map(RunIterations,runList)
    #print (f"Total Jobs= {NTraj}, Jobs Per CPU= {NrunsCPU}, Output Shape = {np.shape(result)}, Job IDs = {runList}")
    for i in range(len(runList)):
        for j in range(NStates):
            for k in range(NStates):
                for l in range(NSteps):
                    rho_ensemble[j,k,l] += result[i][j][k][l]
    print (f"Initial Populations: {rho_ensemble[0,0,0].real} ")

stop = time.time()
print (f"Total Time: {stop - start}")


file05 = open("output_new_rho.txt","w")
for t in range(NSteps):
    file05.write(str(t) + "\t")
    for i in range(NStates):
        file05.write(str(rho_ensemble[i,i,t].real / NTraj) + "\t")
    file05.write("\n")
file05.close()




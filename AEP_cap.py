# %%
import numpy as np
import numba
import matplotlib.pyplot as plt
import pandas as pd


# %%
def grid(h=0.1):  # h in mm
    nz = int(50 / h) + 1    # -Zmax to +Zmax (0 - + 25 mm)
    nr = int(20 / h) + 1    # 0 to Rmax (20 mm)
    
    zvals = np.linspace(-25, 25, nz)
    rvals = np.linspace(0, 20, nr)
    
    return nz, nr, zvals, rvals
    

# %%
def init(phi,zvals, h):
    #(z,r)
    R1_idx = int(3.0 / h)    # R1 = 3mm
    R2_idx = int(5.0 / h)    # R2 = 5mm
    R3_idx = int(7.0 / h)    # R3 = 7mm
    
    # Z idx (need offset for -25 to +25)
    z0 = zvals[0]  # z0 = -25 (first point)
    Z1_idx = int((3.5 - z0) / h)   # Z1 = +3.5mm 
    Z2_idx = int((8.0 - z0) / h)   # Z2 = +8mm  
    Z3_idx = int((15.0 - z0) / h)  # Z3 = +15mm
    Z0_idx = int((0.0 - z0) / h)   # point on the axis (z=0) (in the middle)
    
    # Negative Z 
    nZ1_idx = int((-3.5 - z0) / h)  # -Z1 = -3.5mm 
    nZ2_idx = int((-8.0 - z0) / h)  # -Z2 = -8mm 
    nZ3_idx = int((-15.0 - z0) / h) # -Z3 = -15mm
    
    phi[:, :] = 0.0 #init zero for everything
    # this includes z = -25 vertical line, z = +25 vertical line, r = 20 horizontal line
    # special case for r = 0 horizontal line / see relax function
    

    for zIdx in range(nZ3_idx,nZ1_idx + 1): #cylinder 1
        phi[zIdx,R1_idx] = 0

    for zIdx in range(Z1_idx,Z3_idx + 1): #cylinder 2
        phi[zIdx,R2_idx] = 0

    for zIdx in range(nZ2_idx,Z2_idx + 1): #cylinder 3
        phi[zIdx,R3_idx] = 1000

    for rIdx in range(R1_idx,R3_idx + 1): #disk 1
        phi[Z0_idx,rIdx] = 1000


    return phi


# %%
def BC(nz, nr,zvals,h):
    edges = np.zeros((nz, nr), dtype=np.bool_)  # Initialize all False
    # Edges always fixed at 0V 
    edges[0, :] = 1      # z = -25mm left vertical line
    edges[-1, :] = 1     # z = +25mm top horizontal line
    edges[:, -1] = 1     # r = 20mm right vertical line

    #(z,r)
    R1_idx = int(3.0 / h)    # R1 = 3mm
    R2_idx = int(5.0 / h)    # R2 = 5mm
    R3_idx = int(7.0 / h)    # R3 = 7mm
    
    # Z idx (need offset for -25 to +25)
    z0 = zvals[0]  # z0 = -25 (first point)
    Z1_idx = int((3.5 - z0) / h)   # Z1 = +3.5mm 
    Z2_idx = int((8.0 - z0) / h)   # Z2 = +8mm  
    Z3_idx = int((15.0 - z0) / h)  # Z3 = +15mm
    Z0_idx = int((0.0 - z0) / h)   # point on the axis (z=0) (in the middle)
    
    # Negative Z 
    nZ1_idx = int((-3.5 - z0) / h)  # -Z1 = -3.5mm 
    nZ2_idx = int((-8.0 - z0) / h)  # -Z2 = -8mm 
    nZ3_idx = int((-15.0 - z0) / h) # -Z3 = -15mm
    

    for zIdx in range(nZ3_idx,nZ1_idx + 1): #cylinder 1
        edges[zIdx,R1_idx] = 1

    for zIdx in range(Z1_idx,Z3_idx + 1): #cylinder 2
        edges[zIdx,R2_idx] = 1

    for zIdx in range(nZ2_idx,Z2_idx + 1): #cylinder 3
        edges[zIdx,R3_idx] = 1

    for rIdx in range(R1_idx,R3_idx + 1): #disk 1
        edges[Z0_idx,rIdx] = 1


    return edges

# %%
@numba.jit(nopython=True)
def relax(phi,edges, w=1.0):
    nz, nr = phi.shape
    maxDelta = 0
    for i in range(1,nz-1): #z // stop before boundaries
        for j in range(nr-1): #r // stop before boundaries

            if edges[i,j]:
                continue #skip if on the boundaries
            
            temp = phi[i,j] # phi old // temporary variable

            if j > 0:
                FD = 0.25*(phi[i,j+1] + phi[i,j-1] + phi[i+1,j] + phi[i-1,j]) + (1/(8*j))*(phi[i,j+1] - phi[i,j-1]) # From hint 1 // j > 0
                
            else:
                FD = (1/6)*(4*phi[i,1] + phi[i+1,0] + phi[i-1,0]) # hint 1 // j = 0!
            
            new = (1 - w)*(temp) + w*FD # phi new
            phi[i,j] = new # set phi to phi new

            maxDelta = max(maxDelta, abs(FD - temp)) # see step 4 lecture 7 of the algorithm for relaxtion of FD
        
    return maxDelta


# %%
def sim(h):
    nz, nr, zvals, rvals = grid(h)
    phi = np.zeros((nz,nr), dtype=np.float32)
    phi = init(phi,zvals,h)
    edges = BC(nz,nr,zvals,h)
    return phi,edges,zvals, rvals

# %%
def optOmega():
    h = 0.1 # h = 0.1mm
    tol = 1 #we want to minimize the iterations it takes to reduce the voltage by 1 V

    omega = []
    #omega = [1,1.2,1.4,1.6,1.8,1.85,1.875,1.9,1.925,1.95,1.99] #omega between 1 and ~2. found that 1.8 and 1.9 have the quickest convergence
    for j in range(100): 
        dub = 1.8 + (j/1000) 
        omega.append(dub)
        
    reqInt = [] #required interations needed to find a solution
    for w in omega:
        V, edges, zvals, rvals = sim(h)
        maxIter = 10000
        iter = 0
        
        for i in range(maxIter):
            maxDelta = relax(V,edges,w)
            iter = i
            if (maxDelta < tol):
                print(f"\nSolution found after {i} iterations")
                break
        reqInt.append(iter)

    minIter = np.min(reqInt)
    optW_idx = np.argmin(reqInt) 
    optW = omega[optW_idx]
    print(f"\nOptimal Relaxation Parameter: {optW} at {minIter} Iterations")


    plt.figure()
    plt.plot(omega, reqInt)
    plt.xlabel("Relaxation Parameter")
    plt.ylabel("Iterations Needed")
    plt.title("Relaxation Parameter Optimization")
    plt.savefig('Optimal2.png')
    plt.show()

    res_T = np.array([omega, reqInt]).T
    df = pd.DataFrame(res_T, columns=['Omega', 'Required Iterations'])
    excel = 'optimalOmega2.csv'
    df.to_csv(excel, index=False) 
        
    #optimal omega found to be omega = 1.872
    
    return omega,reqInt

# %%
def capCal(V,zvals,rvals, h):

    # change from mm to m to deal with epsilson
    nz, nr = V.shape
    h_m = h / 1000 # mm -> m

    Ez_m = -np.gradient(V,h_m, axis=0) # delta V / delta z (V/m) column by column (z!)
    Er_m = -np.gradient(V,h_m, axis=1) # delta V / delta r (V/m) row by row (r!)


    E = Ez_m**2 + Er_m**2 # |E| = |Er^2 + Ez^2|
    epsilon = 8.8542e-12 #Farads/meters
    energy = 0.0

    rvals_m = rvals / 1000 # mm -> m

    nz, nr = V.shape
    # integral
    for i in range(1, nz-1):      # don't include boundaries
        for j in range(1, nr-1):
            r = rvals_m[j]
            dr = h_m
            dz = h_m  
            dv = 2 * np.pi * r * dr * dz  # volume element
            energy += E[i, j] * dv

    
    energy = 0.5 * epsilon * energy
    cap = 2 * energy / (1000**2)
    cap_pF = cap * 1e12
    return cap_pF


# %%
def main():
    h = 0.1 #change depending on how precise you want to be -> smaller = more precise
    V, edges, zvals, rvals  = sim(h)

    #relaxation (over = w > 1 ; under = w < 1)
    w = 1.872
    tol = 0.001 
    maxIter = 1000000

    for i in range(maxIter):
        maxDelta = relax(V,edges,w)

        if ((i % 10000) == 0):
             print(f"\nInteration Number = {i} ; Max Delta = {maxDelta}")
        
        if (maxDelta < tol):
            print(f"\nSolution found after {i} iterations")
            break


    # Calculate capacitance
    capacitance = capCal(V, zvals, rvals, h)
    print(f"Capacitance: {capacitance:.9f} pF")

    return V,edges,zvals,rvals,capacitance


# %%
def contour(V,edges,zvals,rvals):

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    cs = ax.contour(zvals, rvals, np.transpose(V)) # ax.contour(y, x, func)
    plt.clabel(cs)
    plt.xlabel("z (mm)")
    plt.ylabel("r (mm)")
    plt.title("Potential V")
    plt.savefig('V.png')
    plt.show()

    return None


# %%
def vary():
    res = []
    hvals = [1.0, 0.5, 0.1, 0.05, 0.025] 
    tolerance = [1.0, 0.1, 0.01, 0.001]

    iter = 1000000
    for h in hvals:
        for tol in tolerance:
            V,edges,zvals, rvals = sim(h)

            for i in range(iter):
                maxDelta = relax(V,edges,w = 1.872) #most optimal

                if ((i % 10000) == 0):
                    print(f"\nInteration Number = {i} ; Max Delta = {maxDelta}")
                if maxDelta < tol:
                    break

            # take the zvals, divide by 2, and round DOWN (this is what the // operator does). It is a floor operator
            center = V[len(zvals)//2, 0] 
            res.append((h,tol, center))
            print(f"\nStep = {h} ; tolerance = {tol} ; Center = {center}")



    df = pd.DataFrame(res, columns=['h (mm)', 'Tolerance (V)', 'V(0,0) (Volts)'])
    csv = 'accuracy2.csv'
    df.to_csv(csv, index=False) 
    return res


# %%
def plot1(V,zvals,rvals,h):

    midpt = len(zvals)//2
    z10 = int(10/h)

    start = midpt-z10
    end = midpt+z10
    zvals = zvals[start:end]


    r = 0
    V_r0 = V[start:end,r]
    Ez_r0 = -np.gradient(V_r0, h)

    plt.figure()
    plt.plot(zvals, V_r0)
    plt.xlabel("z (mm)")
    plt.ylabel("Potential (V)")
    plt.title(f"Potential along r = {r}")
    plt.savefig("potentialr0.png")
    plt.show()

    plt.figure()
    plt.plot(zvals, Ez_r0)
    plt.xlabel("z (mm)")
    plt.ylabel("E_z (V/mm)")
    plt.title(f"E Field along r = {r}")
    plt.savefig("Efieldr0.png")
    plt.show()
    
    return None


# %%
def plot2(V,zvals,rvals,h):
    midpt = len(zvals)//2
    z10 = int(10/h)

    start = midpt-z10
    end = midpt+z10
    zvals = zvals[start:end]

    r = 2.5 # mm
    r_idx = int(r/h)

    V_r25 = V[start:end,r_idx]
    Ez = -np.gradient(V, h, axis=0)  # E_z at r=2.5mm (V/mm)
    Er = -np.gradient(V, h, axis=1)  # E_r at r=2.5mm (V/mm)

    Ez_r25 = Ez[start:end, r_idx]
    Er_r25 = Er[start:end, r_idx]


    plt.figure()
    plt.plot(zvals, V_r25)
    plt.xlabel("z (mm)")
    plt.ylabel("Potential (V)")
    plt.title(f"Potential along r = {r}mm")
    plt.savefig("potentialr25.png")
    plt.show()

    plt.figure()
    plt.plot(zvals,Ez_r25)
    plt.xlabel("z (mm)")
    plt.ylabel(f"E_z at r = {r}mm (V/mm)")
    plt.title(f"Axial E Field along r = {r}")
    plt.savefig("Ezfieldr25.png")
    plt.show()

    plt.figure()
    plt.plot(zvals, Er_r25)
    plt.xlabel("z (mm)")
    plt.ylabel(f"E_r at r = {r}mm (V/mm)")
    plt.title(f"Radial E Field along r = {r}")
    plt.savefig("Erfieldr25.png")
    plt.show()
    
    return None

# %%
# Run the whole thing
#V,edges,zvals,rvals,capacitance = main()
#omega, reqInt = optOmega() # run once
vary() # run once

#plots
#contour(V,edges,zvals,rvals)
#plot1(V,zvals, rvals, h=0.1)    # Fields along r=0
#plot2(V,zvals, rvals, h=0.1)    # Fields along r=2.5mm






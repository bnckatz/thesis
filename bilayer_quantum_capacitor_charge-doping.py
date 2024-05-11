from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure import dos
import matplotlib.pyplot as plt
import os
from pymatgen.io.vasp.inputs import Poscar
import numpy as np

#autoscale function taken from Dan Hickstein's answer at https://stackoverflow.com/a/35094823
def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(0,top)

os.chdir('File directory')
v = Vasprun("vasprun file")
cdos = v.complete_dos
pdos = v.complete_dos.pdos
structure = v.final_structure
poscar=Poscar(structure)


cdos.efermi=cdos.efermi+0.35
d=7
d_2=d*10**(-10)
q_e=1.6*(10**(-19))
m_to_nm=10**9
cell_area=(5.18+2.71)*(4.41+0.15)*(10**(-20))
e_r=1
epsilon_0=8.9*(10**(-12))
const=1/(m_to_nm*epsilon_0*20)

ados = [[i for i in range(len(structure))] for j in range(len(structure))]
for i in range(len(structure)):
    ados[i][0] = cdos.get_site_dos(structure[i]).energies - cdos.efermi
for i in range(len(structure)):
    ados[i][1] = list(cdos.get_site_dos(structure[i]).densities.values())[0]
    ados[i][2] = list(cdos.get_site_dos(structure[i]).densities.values())[1]
edos = cdos.get_element_dos()
plotter = DosPlotter(stack=True)
plotter.add_dos_dict(edos)
plotter.show()

fermi_i=0
i=0
while ados[0][0][i]<0:
    i=i+1
else:
    fermi_i=i-1

num=100
voltages=np.linspace(-7,3.01,num)
sigma_0=(np.array(voltages)*0.41+0.16)/const
ga_fermi=np.zeros((num,1))
gr_fermi=np.zeros((num,1))
j=0
sumga=0
sumgr=0
#sum total gallium charges below fermi level
for i in range(len(ados[0][0])-1):
    if ados[0][0][i]<0:
        for k in range(1, 6):
            sumga=sumga+(ados[len(structure)-k][1][i]+ados[len(structure)-k][2][i])*(ados[0][0][i+1]-ados[0][0][i])/cell_area
#sum total graphene charges below fermi level
for i in range(len(ados[0][0])-1):
    if ados[0][0][i]<0:
        for k in range(1, 16):
            sumgr=sumgr+(ados[len(structure)-k-6][1][i]+ados[len(structure)-k-6][2][i])*(ados[0][0][i+1]-ados[0][0][i])/cell_area

gacharg_net=np.zeros((len(ados[0][0])-1,1))
grcharg_net=np.zeros((len(ados[0][0])-1,1))
gacharg_unit=np.zeros((len(ados[0][0])-1,1))
grcharg_unit=np.zeros((len(ados[0][0])-1,1))
ga_ids=np.zeros((len(ados[0][0])-1,1))
gr_ids=np.zeros((len(ados[0][0])-1,1))
temp=np.zeros((len(ados[0][0])-1,1))
#gr/gacharg_net: # of states/m^2 across energy range from fermi level to ados[0][0][i]
#gr/gacharg_unit: # states/m^2/J at level ados[0][0][i]

#get gallium charges across range
j=0
charg=0
charg2=0
for i in range(len(ados[0][0])-1):
    temp[i]=ados[0][0][i]
    for k in range(1, 6):
        gacharg_unit[j] = ((ados[len(structure)-k][1][i]+ados[len(structure)-k][2][i])/cell_area)/q_e+gacharg_unit[j]
    charg2=charg2+gacharg_unit[j]*(ados[0][0][i+1]-ados[0][0][i])*q_e
    if (ados[0][0][i]<=0):
        gacharg_net[j] = charg2-sumga
        j=j+1
    elif (0<=ados[0][0][i]):
        for k in range(1, 6):
            gacharg_unit[j]=((ados[len(structure)-k][1][i]+ados[len(structure)-k][2][i])/cell_area)/q_e+gacharg_unit[j]
        charg=charg+gacharg_unit[j]*(ados[0][0][i+1]-ados[0][0][i])*q_e
        gacharg_net[j]=charg
        j=j+1
#get graphene charges across range
j=0
charg=0
charg2=0
for i in range(len(ados[0][0])-1):
    for k in range(1, 16):
        grcharg_unit[j] = ((ados[len(structure)-k-6][1][i]+ados[len(structure)-k-6][2][i])/cell_area)/q_e+grcharg_unit[j]
    charg2=charg2+grcharg_unit[j]*(ados[0][0][i+1]-ados[0][0][i])*q_e
    if (ados[0][0][i]<=0):
        grcharg_net[j] = charg2-sumgr
        j=j+1
    elif (0<=ados[0][0][i]):
        for k in range(1, 16):
            grcharg_unit[j]=((ados[len(structure)-k-6][1][i]+ados[len(structure)-k-6][2][i])/cell_area)/q_e+grcharg_unit[j]
        charg=charg+grcharg_unit[j]*(ados[0][0][i+1]-ados[0][0][i])*q_e
        grcharg_net[j]=charg
        j=j+1


for k in [list of distances]:
    d=k
    d_2=d*10**(-10)
    for k2 in [list of epsilon_CHet]:
        e_r=k2
        #do charge-matching
        i=0
        for i in range(len(voltages)):
            gaint = fermi_i
            grint = fermi_i
            ga_energy = 0
            gr_energy = 0
            #iterate through induced charges (one-to-one match with voltages)
            if sigma_0[i]<=0:
                #if the charges don't sum to the induced charge, add more charge
                while np.abs(gacharg_net[gaint]*q_e+grcharg_net[grint]*q_e)<np.abs(sigma_0[i]):
                    #check the energy of added charge for gallium/graphene, and add charge to whichever can do it for the lowest
                    #  energy cost
                    #except instead we actually compare adding the *gallium* charge to the graphene as well, with the following
                    #   reasoning: if, when comparing adding an infinitesimal charge dq to both surfaces, the graphene is preferred,
                    #   the graphene will *continue* to be preferred until its fermi level changes, as the net energy cost of adding
                    #   another dq of charge is the same for the same fermi level (this is *not* true for adding it to the gallium,
                    #   due to the additional energy added in the electric field).

                    addgr=((np.abs(gacharg_net[gaint]*q_e))**2)*d_2/(2*e_r*epsilon_0)+ga_energy+gr_energy+\
                          np.abs(gacharg_unit[gaint-1])*((np.abs(ados[0][0][grint])+np.abs(ados[0][0][grint-1]))/2)*q_e*(ados[0][0][gaint]-ados[0][0][gaint-1])*q_e
                    addga=(((np.abs(gacharg_net[gaint])+np.abs(gacharg_unit[gaint-1])*(ados[0][0][gaint]-ados[0][0][gaint-1])*q_e)*q_e)**2)*d_2/(2*e_r*epsilon_0)+ga_energy+gr_energy+\
                          np.abs(gacharg_unit[gaint-1])*((np.abs(ados[0][0][gaint])+np.abs(ados[0][0][gaint-1]))/2)*q_e*(ados[0][0][gaint]-ados[0][0][gaint-1])*q_e
                    if addgr<addga:
                        gr_energy=gr_energy+np.abs(grcharg_unit[grint-1])*((np.abs(ados[0][0][grint])+np.abs(ados[0][0][grint-1]))/2)*q_e*(ados[0][0][grint]-ados[0][0][grint-1])*q_e
                        grint=grint-1
                    else:
                        ga_energy=ga_energy+np.abs(gacharg_unit[gaint-1])*((np.abs(ados[0][0][gaint])+np.abs(ados[0][0][gaint-1]))/2)*q_e*(ados[0][0][gaint]-ados[0][0][gaint-1])*q_e
                        gaint=gaint-1
            elif sigma_0[i]>=0:
                while np.abs(gacharg_net[gaint]*q_e+grcharg_net[grint]*q_e)<np.abs(sigma_0[i]):
                    addgr=((np.abs(gacharg_net[gaint]*q_e))**2)*d_2/(2*e_r*epsilon_0)+\
                          np.abs(gacharg_unit[gaint+1])*((np.abs(ados[0][0][grint])+np.abs(ados[0][0][grint+1]))/2)*q_e*(ados[0][0][gaint+1]-ados[0][0][gaint])*q_e
                    addga=(((np.abs(gacharg_net[gaint])+np.abs(gacharg_unit[gaint+1])*(ados[0][0][gaint+1]-ados[0][0][gaint])*q_e)*q_e)**2)*d_2/(2*e_r*epsilon_0)+\
                          np.abs(gacharg_unit[gaint+1])*((np.abs(ados[0][0][gaint])+np.abs(ados[0][0][gaint+1]))/2)*q_e*(ados[0][0][gaint+1]-ados[0][0][gaint])*q_e
                    if addgr<addga:
                        gr_energy=gr_energy+np.abs(grcharg_unit[grint+1])*((np.abs(ados[0][0][grint])+np.abs(ados[0][0][grint+1]))/2)*q_e*(ados[0][0][grint+1]-ados[0][0][grint])*q_e
                        grint=grint+1
                    else:
                        ga_energy=ga_energy+np.abs(gacharg_unit[gaint+1])*((np.abs(ados[0][0][gaint])+np.abs(ados[0][0][gaint+1]))/2)*q_e*(ados[0][0][gaint+1]-ados[0][0][gaint])*q_e
                        gaint=gaint+1

            #assign fermi levels based on whichever fermi level indexes gave the right amount of charge
            ga_fermi[i] = ados[0][0][gaint]
            gr_fermi[i] = ados[0][0][grint]
            ga_ids[i]=gaint
            gr_ids[i]=grint
            i = i + 1

        plt.plot(voltages,ga_fermi)
        plt.plot(voltages,gr_fermi)
        plt.xticks([-7,-6,-5,-4,-3,-2,-1,0,1,2,3])
        plt.ylabel("Fermi level (eV)")
        plt.xlabel("Input Voltage (V)")
        plt.grid("both")
        plt.show()
quit()

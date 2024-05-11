import numpy as np
import copy
from lammps import lammps


# from mpi4py import MPI

def find_item(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return mylist.index(sub_list), sub_list.index(char)
    else:
        return -1


lmp = lammps()
lmp.file('lmp_header.txt')
f = open('status.txt', 'w')
f2 = open('atoms_list.txt', 'w')

d1 = 6
d2 = 6
temp = 800
kbT = (8.617333262145 * (10 ** (-5))) * (temp + 273.15)
minsize = 5
maxsize = 8

bonds = [[[0 for l in range(0, 3)], [0 for m in range(0, 3)]] for i in range(0, 3 * d1 * d2)]
rings = [[] for i in range(0, d1 * d2)]
particles = [[[0 for k in range(0, 2)] for j in range(0, 3)] for i in range(0, 2 * d1 * d2)]
prings = [[0 for j in range(0, 3)] for i in range(0, 2 * d1 * d2)]
# rings is just an index of how many sides each ring has, these all start with six
# particles is a correspondence table to say which bonds a particle is in:
#   particles[particle ID]=[3x[bond, particle it's bound to]]

# set up bonds: bonds[bond ID]=[rings],[particles]: [rings]=[side1,side2,end1,end2] rings around a bond
#      [particles]=[ID1,n1,n2],[ID2,n1,n2] IDs of particles in bond and their two neighbors, ordered properly
for k in range(0, d2):
    for i in range(0, d1):
        # set up bonds: set which atoms are initially at the ends of which bond, and which atoms
        #  they're bonded to (in clockwise order, given a facing with low-numbered atoms proceeding)
        #  top-left to top-right, then downwards
        bonds[3 * (i + d1 * k)][0][0] = np.mod(2 * i, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k)][0][1] = np.mod(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * (k - 1), 2 * d1 * d2)
        bonds[3 * (i + d1 * k)][0][2] = np.mod(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k, 2 * d1 * d2)
        bonds[3 * (i + d1 * k)][1][0] = np.mod(2 * i - 1, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k)][1][1] = np.mod(np.mod(2 * i - 2, 2 * d1) + 2 * d1 * (k + 1), 2 * d1 * d2)
        bonds[3 * (i + d1 * k)][1][2] = np.mod(2 * i - 2, 2 * d1) + 2 * d1 * k

        bonds[3 * (i + d1 * k) + 1][0][0] = np.mod(2 * i, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k) + 1][0][1] = np.mod(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k, 2 * d1 * d2)
        bonds[3 * (i + d1 * k) + 1][0][2] = np.mod(2 * i - 1, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k) + 1][1][0] = np.mod(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * (k - 1), 2 * d1 * d2)
        bonds[3 * (i + d1 * k) + 1][1][1] = np.mod(np.mod(2 * i, 2 * d1) + 2 * d1 * (k - 1), 2 * d1 * d2)
        bonds[3 * (i + d1 * k) + 1][1][2] = np.mod(np.mod(2 * i + 2, 2 * d1) + 2 * d1 * (k - 1), 2 * d1 * d2)

        bonds[3 * (i + d1 * k) + 2][0][0] = np.mod(2 * i, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k) + 2][0][1] = np.mod(2 * i - 1, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k) + 2][0][2] = np.mod(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * (k - 1), 2 * d1 * d2)
        bonds[3 * (i + d1 * k) + 2][1][0] = np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k) + 2][1][1] = np.mod(2 * i + 2, 2 * d1) + 2 * d1 * k
        bonds[3 * (i + d1 * k) + 2][1][2] = np.mod(np.mod(2 * i, 2 * d1) + 2 * d1 * (k + 1), 2 * d1 * d2)

        # assign bonds as -[bond ID]-1, so that they're distinguishable from atom IDs
        particles[np.mod(2 * i, 2 * d1) + 2 * d1 * k][0] = \
            [-3 * (i + d1 * k) - 1, bonds[3 * (i + d1 * k)][1][0]]
        particles[np.mod(2 * i - 1, 2 * d1) + 2 * d1 * k][0] = \
            [-3 * (i + d1 * k) - 1, bonds[3 * (i + d1 * k)][0][0]]
        particles[np.mod(2 * i, 2 * d1) + 2 * d1 * k][1] = \
            [-3 * (i + d1 * k) - 1 - 1, bonds[3 * (i + d1 * k) + 1][1][0]]
        particles[np.mod(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * (k - 1), 2 * d1 * d2)][1] = \
            [-3 * (i + d1 * k) - 1 - 1, bonds[3 * (i + d1 * k) + 1][0][0]]
        particles[np.mod(2 * i, 2 * d1) + 2 * d1 * k][2] = \
            [-3 * (i + d1 * k) - 2 - 1, bonds[3 * (i + d1 * k) + 2][1][0]]
        particles[np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k][2] = \
            [-3 * (i + d1 * k) - 2 - 1, bonds[3 * (i + d1 * k) + 2][0][0]]

        # set up rings around atoms, clockwise
        prings[2 * i + 2 * d1 * k] = [i + d1 * k, np.mod(i + d1 * (k - 1), d1 * d2),
                                      np.mod(np.mod(i + 1, d1) + d1 * (k - 1), d1 * d2)]
        rings[i + d1 * k].append(2 * i + 2 * d1 * k)
        rings[np.mod(i + d1 * (k - 1), d1 * d2)].append(2 * i + 2 * d1 * k)
        rings[np.mod(np.mod(i + 1, d1) + d1 * (k - 1), d1 * d2)].append(2 * i + 2 * d1 * k)

        prings[np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k] = \
            [i + d1 * k, np.mod(np.mod(i + 1, d1) + d1 * (k - 1), d1 * d2),
             np.mod(np.mod(i + 1, d1) + d1 * k, d1 * d2)]
        rings[i + d1 * k].append(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k)
        rings[np.mod(np.mod(i + 1, d1) + d1 * (k - 1), d1 * d2)].append(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k)
        rings[np.mod(np.mod(i + 1, d1) + d1 * k, d1 * d2)].append(np.mod(2 * i + 1, 2 * d1) + 2 * d1 * k)

lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('minimize 1e-25 1e-25 10000 10000')
old_energy = lmp.get_thermo('pe')
lmp.command('pair_style zero 16.0 nocoeff')
lmp.command('pair_coeff * * CH.airebo C')
lmp.command('fix fixbonds all bond/create 1 1 1 2.2 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
lmp.command('run 200')
lmp.command('unfix fixbonds')
lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('unfix press')
lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('minimize 1e-25 1e-25 10000 10000')
lmp.command('undump 1')
lmp.command('undump s')
lmp.command('displace_atoms all random 0 0 0.1 1234523')
nbonds = lmp.gather_bonds()[0]

k = 0
while k < 50:
    bond = int(np.random.rand(1)[0] * 3 * d1 * d2)
    # find common pairs of rings in the three rings surrounding the atoms selected by this bond
    a1 = bonds[bond][0][0]
    a2 = bonds[bond][1][0]
    t1 = np.array((prings[a1]))
    t2 = np.array((prings[a2]))
    common_rings = np.intersect1d(t1, t2)
    if len(common_rings) == 3:
        t1[2] = -3
    # find rings which aren't shared: each atom enters the ring it doesn't share with the other atom
    r_add_a2 = np.setdiff1d(t1, t2)[0]
    r_add_a1 = np.setdiff1d(t2, t1)[0]
    # get indexes of common rings: we rotate clockwise, and the rings are stored clockwise, so by looking
    #   at the indexes of the common rings, we can identify the ring most counterclockwise of
    #   the pair of rings relative to a given atom: this is the ring that atom leaves
    a1_i = [prings[a1].index(common_rings[0]), prings[a1].index(common_rings[1])]
    a2_i = [prings[a2].index(common_rings[0]), prings[a2].index(common_rings[1])]

    if a1_i[1] == a1_i[0] + 1 or a1_i[1] == a1_i[0] - 2:
        r_leave_a1 = prings[a1][a1_i[1]]
    else:
        r_leave_a1 = prings[a1][a1_i[0]]
    if a2_i[1] == a2_i[0] + 1 or a2_i[1] == a2_i[0] - 2:
        r_leave_a2 = prings[a2][a2_i[1]]
    else:
        r_leave_a2 = prings[a2][a2_i[0]]

    if len(rings[r_leave_a1]) - 1 != minsize - 1 and len(rings[r_leave_a2]) - 1 != minsize - 1 \
            and len(rings[r_add_a1]) + 1 != maxsize + 1 and len(rings[r_add_a2]) + 1 != maxsize + 1 \
            and len(common_rings) < 3:
        k = k + 1
        print(k, file=f)
        p11 = bonds[bond][0][1]
        p12 = bonds[bond][0][2]
        p21 = bonds[bond][1][1]
        p22 = bonds[bond][1][2]
        p1 = bonds[bond][0][0]
        p2 = bonds[bond][1][0]

        # DEAL WITH THE FACT THAT BOND 0 AND PARTICLE 0 AREN'T THE SAME
        b11 = -particles[p11][find_item(particles[p11], p1)[0]][0] - 1
        b21 = -particles[p21][find_item(particles[p21], p2)[0]][0] - 1
        b12 = -particles[p12][find_item(particles[p12], p1)[0]][0] - 1
        b22 = -particles[p22][find_item(particles[p22], p2)[0]][0] - 1

        # copy ring/bond/particle info, in case we have to reject the rotation
        rstore = [copy.deepcopy(rings[r_leave_a1]), copy.deepcopy(rings[r_leave_a2]),
                  copy.deepcopy(rings[r_add_a1]), copy.deepcopy(rings[r_add_a2]),
                  copy.deepcopy(prings[a1]), copy.deepcopy(prings[a2]),
                  copy.deepcopy(bonds[bond]), copy.deepcopy(bonds[b11]), copy.deepcopy(bonds[b12]),
                  copy.deepcopy(bonds[b21]), copy.deepcopy(bonds[b22]), copy.deepcopy(particles[p1]),
                  copy.deepcopy(particles[p2]), copy.deepcopy(particles[p11]), copy.deepcopy(particles[p12]),
                  copy.deepcopy(particles[p21]), copy.deepcopy(particles[p22])]

        rings[r_leave_a1].remove(a1)
        rings[r_leave_a2].remove(a2)
        rings[r_add_a1].append(a1)
        rings[r_add_a2].append(a2)
        prings[a1][prings[a1].index(r_leave_a1)] = r_add_a1
        prings[a2][prings[a2].index(r_leave_a2)] = r_add_a2

        bonds[bond][0][1] = bonds[bond][0][2]
        bonds[bond][0][2] = bonds[bond][1][1]
        bonds[bond][1][1] = bonds[bond][1][2]
        bonds[bond][1][2] = p11

        # update the bonds connected to this bond: these are bonds b111,b112,b211,b212
        store = list(range(0, 3))
        store.remove(find_item(particles[p11], p1)[0])
        for i in store:
            b = -particles[p11][i][0] - 1
            bi = find_item(bonds[b], p11)[0]
            pi = bonds[b][bi].index(p1)
            bonds[b][bi][pi] = p2
            rstore.append([b, bi, pi])

        store = list(range(0, 3))
        store.remove(find_item(particles[p21], p2)[0])
        for i in store:
            b = -particles[p21][i][0] - 1
            bi = find_item(bonds[b], p21)[0]
            pi = bonds[b][bi].index(p2)
            bonds[b][bi][pi] = p1
            rstore.append([b, bi, pi])

        bonds[b11][find_item(bonds[b11], p1)[0]] = [p2, p1, p22]
        bonds[b12][find_item(bonds[b12], p1)[0]] = [p1, p21, p2]
        bonds[b21][find_item(bonds[b21], p2)[0]] = [p1, p2, p12]
        bonds[b22][find_item(bonds[b22], p2)[0]] = [p2, p11, p1]

        particles[p1][find_item(particles[p1], -b11 - 1)[0]] = [-b21 - 1, p21]
        particles[p11][find_item(particles[p11], -b11 - 1)[0]] = [-b11 - 1, p2]
        particles[p2][find_item(particles[p2], -b21 - 1)[0]] = [-b11 - 1, p11]
        particles[p21][find_item(particles[p21], -b21 - 1)[0]] = [-b21 - 1, p1]

        atom1 = p1
        atom2 = p2
        print('----------------------------------------', k, file=f2)
        print(str(p1) + '-' + str(p11) + '->' + str(p1) + '-' + str(p21) + ',',
              str(p2) + '-' + str(p21) + '->' + str(p2) + '-' + str(p11) + ',',
              str(p1) + '-' + str(p12) + ',', str(p2) + '-' + str(p22), file=f2)
        try:

            # save pre-rotated state in case rotation is rejected
            lmp.command('write_restart restart.rstrt')
            # get atom positions, shift so that two atoms to rotate are in the center
            # otherwise they may overlie a periodic boundary and will rotate incorrectly
            lmp.command('variable dx equal $(x[' + str(atom1 + 1) + '])')
            lmp.command('variable dy equal $(y[' + str(atom1 + 1) + '])')
            lmp.command('displace_atoms all move $(-v_dx+(xhi-xy-xlo)/2) $(-v_dy+(yhi-ylo)/2) 0')
            lmp.command('run 1')
            lmp.command('variable a1x equal $(x[' + str(atom1 + 1) + '])')
            lmp.command('variable a1y equal $(y[' + str(atom1 + 1) + '])')
            lmp.command('variable a2x equal $(x[' + str(atom2 + 1) + '])')
            lmp.command('variable a2y equal $(y[' + str(atom2 + 1) + '])')
            lmp.command('group set1 id {}'.format(atom1 + 1))
            lmp.command('group set2 id {}'.format(atom2 + 1))
            # rotate the atoms by manually setting their new coordinates
            #   just using displace_atoms rotate *should* work for this, but it doesn't.
            #   for some reason, it treats them as if they're in some further periodic image and
            #   rotates them incorrectly as a result
            lmp.command('set group set1 x $(((v_a1y-v_a2y)+v_a1x+v_a2x)/2) y $(((v_a2x-v_a1x)+v_a1y+v_a2y)/2)')
            lmp.command('set group set2 x $(((v_a2y-v_a1y)+v_a1x+v_a2x)/2) y $(((v_a1x-v_a2x)+v_a1y+v_a2y)/2)')
            lmp.command('displace_atoms all move $(v_dx-(xhi-xy-xlo)/2) $(v_dy-(yhi-ylo)/2) 0')
            # identify atoms to break bonds/form bonds, then break and reform new bonds to new neighbors
            lmp.command('group break1 id {} {}'.format(atom1 + 1, p11 + 1))
            lmp.command('group break2 id {} {}'.format(atom2 + 1, p21 + 1))
            lmp.command('fix break1 break1 bond/break 1 1 0.1')
            lmp.command('run 20')
            lmp.command('unfix break1')
            lmp.command('fix break2 break2 bond/break 1 1 0.1')
            lmp.command('run 20')
            lmp.command('unfix break2')
            lmp.command('group bond1 id {} {}'.format(atom1 + 1, p21 + 1))
            lmp.command('group bond2 id {} {}'.format(atom2 + 1, p11 + 1))
            # turn off the airebo potential to save calculation time
            lmp.command('pair_style zero 16.0 nocoeff')
            lmp.command('pair_coeff * * CH.airebo C')
            # raise the communication distance just in case it's needed for a bond
            lmp.command('comm_modify cutoff 26')
            lmp.command('fix fixbonds1 bond1 bond/create 1 1 1 8 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
            lmp.command('run 20')
            lmp.command('unfix fixbonds1')
            lmp.command('fix fixbonds2 bond2 bond/create 1 1 1 8 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
            lmp.command('run 20')
            lmp.command('unfix fixbonds2')
            # lower the communication distance because it's not needed now that no new bonds are forming
            lmp.command('comm_modify cutoff 8')
            lmp.command('minimize 1e-25 1e-25 10000 100000')
            # break all bonds before using airebo to minimize
            #   you ought to be able to just turn off the bonded interactions to have the non-bonded potential
            #   by itself, but for some reason that doesn't work and the energies/relaxed positions come out
            #   visibly wrong
            lmp.command('fix break all bond/break 1 1 0.1')
            lmp.command('run 200')
            lmp.command('unfix break')
            # turn on airebo potential to re-relax under the non-bonded potential
            lmp.command('pair_style airebo 4.0')
            lmp.command('pair_coeff * * CH.airebo C')
            lmp.command('minimize 1e-25 1e-25 10000 100000')
            lmp.command('unfix press')
            lmp.command('min_style fire')
            lmp.command('minimize 1e-25 1e-25 1000 10000')

            # get energy for comparison here so we don't neeed to do it in the comparison step
            new_energy = lmp.get_thermo('pe')
            lmp.command('fix fixbonds all bond/create 1 1 1 1.8 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
            lmp.command('run 500')
            lmp.command('unfix fixbonds')
            lmp.command('min_style cg')
            lmp.command('dump 1 all custom 1 smallshapes_periodic_1hex.' + str(k) + ' id type x y z c_eng')
            lmp.command('dump s all local 1 shapebonds.xyz_' + str(k) + ' index c_blist[1] c_blist[2] c_blist[3]')
            lmp.command('run 1')
            lmp.command('undump 1')
            lmp.command('undump s')

            # monte-carlo part: accept or reject only if energy of new state is too much higher than
            #   that of old state, *or* if a bond has broken
            if new_energy>(old_energy+4.5+np.exp(-kbT*np.random.rand())) or lmp.gather_bonds()[0]<nbonds:
            #if lmp.gather_bonds()[0] < nbonds:
                # reset ring counts, neighbors, bonding, etc.:
                rings[r_leave_a1] = rstore[0]
                rings[r_leave_a2] = rstore[1]
                rings[r_add_a1] = rstore[2]
                rings[r_add_a2] = rstore[3]
                prings[a1] = rstore[4]
                prings[a2] = rstore[5]
                bonds[bond] = rstore[6]
                bonds[b11] = rstore[7]
                bonds[b12] = rstore[8]
                bonds[b21] = rstore[9]
                bonds[b22] = rstore[10]
                particles[p1] = rstore[11]
                particles[p2] = rstore[12]
                particles[p11] = rstore[13]
                particles[p12] = rstore[14]
                particles[p21] = rstore[15]
                particles[p22] = rstore[16]
                bonds[rstore[17][0]][rstore[17][1]][rstore[17][2]] = p1
                bonds[rstore[18][0]][rstore[18][1]][rstore[18][2]] = p1
                bonds[rstore[19][0]][rstore[19][1]][rstore[19][2]] = p2
                bonds[rstore[20][0]][rstore[20][1]][rstore[20][2]] = p2
                print('flip rejected, oldE={:.2f}'.format(old_energy), 'newE={:.2f}'.format(new_energy),
                      'nbonds=' + str(lmp.gather_bonds()[0]), file=f)

                # then jump back to before rotation
                lmp.close()
                lmp = lammps()
                lmp.command('box tilt large')
                lmp.command('read_restart restart.rstrt')
                lmp.command('compute eng all pe/atom')
                lmp.command('compute eatoms all reduce sum c_eng')
                lmp.command('compute blist all property/local btype batom1 batom2')
                lmp.command('pair_style zero 16.0 nocoeff')
                lmp.command('pair_coeff * * CH.airebo C')
                lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
            else:
                print('flip accepted, oldE={:.2f}'.format(old_energy), 'newE={:.2f}'.format(new_energy),
                      'nbonds=' + str(lmp.gather_bonds()[0]), file=f)

                lengths = []
                for i in rings:
                    lengths.append(len(i))
                num = [lengths.count(i) for i in range(maxsize + 1)]
                print('Rings of size', end=' ', file=f)
                for i in range(maxsize + 1):
                    print(str(i) + ':', str(num[i]) + ',', end=' ', file=f)
                print('\n', file=f)

                # if not rejected, set new energy of comparison
                old_energy = new_energy
                lmp.command('group break1 delete')
                lmp.command('group break2 delete')
                lmp.command('group bond1 delete')
                lmp.command('group bond2 delete')
                lmp.command('variable dx delete')
                lmp.command('variable dy delete')
                lmp.command('variable a1x delete')
                lmp.command('variable a2x delete')
                lmp.command('variable a1y delete')
                lmp.command('variable a2y delete')
                lmp.command('group set1 delete')
                lmp.command('group set2 delete')
                lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
        except Exception as e:
            print('flip rejected:', e, file=f)
            # reset ring counts, neighbors, bonding, etc.:
            rings[r_leave_a1] = rstore[0]
            rings[r_leave_a2] = rstore[1]
            rings[r_add_a1] = rstore[2]
            rings[r_add_a2] = rstore[3]
            prings[a1] = rstore[4]
            prings[a2] = rstore[5]
            bonds[bond] = rstore[6]
            bonds[b11] = rstore[7]
            bonds[b12] = rstore[8]
            bonds[b21] = rstore[9]
            bonds[b22] = rstore[10]
            particles[p1] = rstore[11]
            particles[p2] = rstore[12]
            particles[p11] = rstore[13]
            particles[p12] = rstore[14]
            particles[p21] = rstore[15]
            particles[p22] = rstore[16]
            bonds[rstore[17][0]][rstore[17][1]][rstore[17][2]] = p1
            bonds[rstore[18][0]][rstore[18][1]][rstore[18][2]] = p1
            bonds[rstore[19][0]][rstore[19][1]][rstore[19][2]] = p2
            bonds[rstore[20][0]][rstore[20][1]][rstore[20][2]] = p2

            # then jump back to before rotation
            lmp.close()
            lmp = lammps()
            lmp.command('box tilt large')
            lmp.command('read_restart restart.rstrt')
            lmp.command('compute eng all pe/atom')
            lmp.command('compute eatoms all reduce sum c_eng')
            lmp.command('compute blist all property/local btype batom1 batom2')
            lmp.command('pair_style zero 16.0 nocoeff')
            lmp.command('pair_coeff * * CH.airebo C')
            lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
add = k
for i in range(0, d2):
    x = [len(rings[j + d1 * i]) for j in range(0, d1)]
    for j in range(0, i):
        print('  ', end='', file=f)
    print(x, file=f)
print("----------------------------------------", file=f)

lengths = []
for i in rings:
    lengths.append(len(i))
num = [lengths.count(i) for i in range(maxsize + 1)]
print('Rings of size', end=' ', file=f)
for i in range(maxsize + 1):
    print(str(i) + ':', str(num[i]) + ',', end=' ', file=f)
print('\n', file=f)

unchangedrotsremaining = np.floor(d1 * d2 * 10)
k = 0
trynumber = 0
while k < 500:
    bond = int(np.random.rand(1)[0] * 3 * d1 * d2)
    if trynumber == 100:
        k = k + 1
        trynumber = 0

    # find common pairs of rings in the three rings surrounding the atoms selected by this bond
    a1 = bonds[bond][0][0]
    a2 = bonds[bond][1][0]
    t1 = np.array((prings[a1]))
    t2 = np.array((prings[a2]))
    common_rings = np.intersect1d(t1, t2)
    if len(common_rings) == 3:
        t1[2] = -3
    # find rings which aren't shared: each atom enters the ring it doesn't share with the other atom
    r_add_a2 = np.setdiff1d(t1, t2)[0]
    r_add_a1 = np.setdiff1d(t2, t1)[0]
    # get indexes of common rings: we rotate clockwise, and the rings are stored clockwise, so by looking
    #   at the indexes of the common rings, we can identify the ring most counterclockwise of
    #   the pair of rings relative to a given atom: this is the ring that atom leaves
    a1_i = [prings[a1].index(common_rings[0]), prings[a1].index(common_rings[1])]
    a2_i = [prings[a2].index(common_rings[0]), prings[a2].index(common_rings[1])]

    if a1_i[1] == a1_i[0] + 1 or a1_i[1] == a1_i[0] - 2:
        r_leave_a1 = prings[a1][a1_i[1]]
    else:
        r_leave_a1 = prings[a1][a1_i[0]]
    if a2_i[1] == a2_i[0] + 1 or a2_i[1] == a2_i[0] - 2:
        r_leave_a2 = prings[a2][a2_i[1]]
    else:
        r_leave_a2 = prings[a2][a2_i[0]]

    t = 0
    check = 0
    if len(rings[r_leave_a1]) - 1 != minsize - 1 and len(rings[r_leave_a2]) - 1 != minsize - 1 \
            and len(rings[r_add_a1]) + 1 != maxsize + 1 and len(rings[r_add_a2]) + 1 != maxsize + 1 \
            and len(common_rings) < 3:
        t = 1
        rstore1 = [copy.deepcopy(rings[r_leave_a1]), copy.deepcopy(rings[r_leave_a2]),
                   copy.deepcopy(rings[r_add_a1]), copy.deepcopy(rings[r_add_a2]),
                   copy.deepcopy(prings[a1]), copy.deepcopy(prings[a2])]
        rings[r_leave_a1].remove(a1)
        rings[r_leave_a2].remove(a2)
        rings[r_add_a1].append(a1)
        rings[r_add_a2].append(a2)
        prings[a1][prings[a1].index(r_leave_a1)] = r_add_a1
        prings[a2][prings[a2].index(r_leave_a2)] = r_add_a2

        lengths = []
        for i in rings:
            lengths.append(len(i))
        numi = [lengths.count(i) for i in range(maxsize + 1)]

        # if num5i < num5 or num6i > num6 or num7i < num7:
        for i in range(minsize,6):
            for j in range(i+1,7):
                if numi[i]<num[i] and numi[j]>num[j]:
                    check=1
        for i in range(6,maxsize+1):
            for j in range(i+1,6):
                if numi[i]>num[i] and numi[j]<num[j]:
                    check=1
        if unchangedrotsremaining > 0 and numi[6] == num[6]:
            unchangedrotsremaining = unchangedrotsremaining - 1
            check = 1

        if check == 1:
            k = k + 1
            print(k + add, file=f)
            p11 = bonds[bond][0][1]
            p12 = bonds[bond][0][2]
            p21 = bonds[bond][1][1]
            p22 = bonds[bond][1][2]
            p1 = bonds[bond][0][0]
            p2 = bonds[bond][1][0]

            # DEAL WITH THE FACT THAT BOND 0 AND PARTICLE 0 AREN'T THE SAME
            b11 = -particles[p11][find_item(particles[p11], p1)[0]][0] - 1
            b21 = -particles[p21][find_item(particles[p21], p2)[0]][0] - 1
            b12 = -particles[p12][find_item(particles[p12], p1)[0]][0] - 1
            b22 = -particles[p22][find_item(particles[p22], p2)[0]][0] - 1

            # copy ring/bond/particle info, in case we have to reject the rotation
            rstore = [rstore1[0], rstore1[1], rstore1[2], rstore1[3], rstore1[4], rstore1[5],
                      copy.deepcopy(bonds[bond]), copy.deepcopy(bonds[b11]), copy.deepcopy(bonds[b12]),
                      copy.deepcopy(bonds[b21]), copy.deepcopy(bonds[b22]), copy.deepcopy(particles[p1]),
                      copy.deepcopy(particles[p2]), copy.deepcopy(particles[p11]), copy.deepcopy(particles[p12]),
                      copy.deepcopy(particles[p21]), copy.deepcopy(particles[p22])]

            bonds[bond][0][1] = bonds[bond][0][2]
            bonds[bond][0][2] = bonds[bond][1][1]
            bonds[bond][1][1] = bonds[bond][1][2]
            bonds[bond][1][2] = p11

            # update the bonds connected to this bond: these are bonds b111,b112,b211,b212
            store = list(range(0, 3))
            store.remove(find_item(particles[p11], p1)[0])
            for i in store:
                b = -particles[p11][i][0] - 1
                bi = find_item(bonds[b], p11)[0]
                pi = bonds[b][bi].index(p1)
                bonds[b][bi][pi] = p2
                rstore.append([b, bi, pi])

            store = list(range(0, 3))
            store.remove(find_item(particles[p21], p2)[0])
            for i in store:
                b = -particles[p21][i][0] - 1
                bi = find_item(bonds[b], p21)[0]
                pi = bonds[b][bi].index(p2)
                bonds[b][bi][pi] = p1
                rstore.append([b, bi, pi])

            bonds[b11][find_item(bonds[b11], p1)[0]] = [p2, p1, p22]
            bonds[b12][find_item(bonds[b12], p1)[0]] = [p1, p21, p2]
            bonds[b21][find_item(bonds[b21], p2)[0]] = [p1, p2, p12]
            bonds[b22][find_item(bonds[b22], p2)[0]] = [p2, p11, p1]

            particles[p1][find_item(particles[p1], -b11 - 1)[0]] = [-b21 - 1, p21]
            particles[p11][find_item(particles[p11], -b11 - 1)[0]] = [-b11 - 1, p2]
            particles[p2][find_item(particles[p2], -b21 - 1)[0]] = [-b11 - 1, p11]
            particles[p21][find_item(particles[p21], -b21 - 1)[0]] = [-b21 - 1, p1]

            atom1 = p1
            atom2 = p2
            print('----------------------------------------', str(k + add), file=f2)
            print(str(p1) + '-' + str(p11) + '->' + str(p1) + '-' + str(p21) + ',',
                  str(p2) + '-' + str(p21) + '->' + str(p2) + '-' + str(p11) + ',',
                  str(p1) + '-' + str(p12) + ',', str(p2) + '-' + str(p22), file=f2)
            try:
                # save pre-rotated state in case rotation is rejected
                lmp.command('write_restart restart.rstrt')
                # get atom positions, shift so that two atoms to rotate are in the center
                # otherwise they may overlie a periodic boundary and will rotate incorrectly
                lmp.command('variable dx equal $(x[' + str(atom1 + 1) + '])')
                lmp.command('variable dy equal $(y[' + str(atom1 + 1) + '])')
                lmp.command('displace_atoms all move $(-v_dx+(xhi-xy-xlo)/2) $(-v_dy+(yhi-ylo)/2) 0')
                lmp.command('run 1')
                lmp.command('variable a1x equal $(x[' + str(atom1 + 1) + '])')
                lmp.command('variable a1y equal $(y[' + str(atom1 + 1) + '])')
                lmp.command('variable a2x equal $(x[' + str(atom2 + 1) + '])')
                lmp.command('variable a2y equal $(y[' + str(atom2 + 1) + '])')
                lmp.command('group set1 id {}'.format(atom1 + 1))
                lmp.command('group set2 id {}'.format(atom2 + 1))
                # rotate the atoms by manually setting their new coordinates
                #   just using displace_atoms rotate *should* work for this, but it doesn't.
                #   for some reason, it treats them as if they're in some further periodic image and
                #   rotates them incorrectly as a result
                lmp.command('set group set1 x $(((v_a1y-v_a2y)+v_a1x+v_a2x)/2) y $(((v_a2x-v_a1x)+v_a1y+v_a2y)/2)')
                lmp.command('set group set2 x $(((v_a2y-v_a1y)+v_a1x+v_a2x)/2) y $(((v_a1x-v_a2x)+v_a1y+v_a2y)/2)')
                lmp.command('displace_atoms all move $(v_dx-(xhi-xy-xlo)/2) $(v_dy-(yhi-ylo)/2) 0')
                # identify atoms to break bonds/form bonds, then break and reform new bonds to new neighbors
                lmp.command('group break1 id {} {}'.format(atom1 + 1, p11 + 1))
                lmp.command('group break2 id {} {}'.format(atom2 + 1, p21 + 1))
                lmp.command('fix break1 break1 bond/break 1 1 0.1')
                lmp.command('run 20')
                lmp.command('unfix break1')
                lmp.command('fix break2 break2 bond/break 1 1 0.1')
                lmp.command('run 20')
                lmp.command('unfix break2')
                lmp.command('group bond1 id {} {}'.format(atom1 + 1, p21 + 1))
                lmp.command('group bond2 id {} {}'.format(atom2 + 1, p11 + 1))
                # turn off the airebo potential to save calculation time
                lmp.command('pair_style zero 16.0 nocoeff')
                lmp.command('pair_coeff * * CH.airebo C')
                # raise the communication distance just in case it's needed for a bond
                lmp.command('comm_modify cutoff 26')
                lmp.command('fix fixbonds1 bond1 bond/create 1 1 1 8 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
                lmp.command('run 20')
                lmp.command('unfix fixbonds1')
                lmp.command('fix fixbonds2 bond2 bond/create 1 1 1 8 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
                lmp.command('run 20')
                lmp.command('unfix fixbonds2')
                # lower the communication distance because it's not needed now that no new bonds are forming
                lmp.command('comm_modify cutoff 8')
                lmp.command('minimize 1e-25 1e-25 10000 100000')
                # break all bonds before using airebo to minimize
                #   you ought to be able to just turn off the bonded interactions to have the non-bonded potential
                #   by itself, but for some reason that doesn't work and the energies/relaxed positions come out
                #   visibly wrong
                lmp.command('fix break all bond/break 1 1 0.1')
                lmp.command('run 200')
                lmp.command('unfix break')
                # turn on airebo potential to re-relax under the non-bonded potential
                lmp.command('pair_style airebo 4.0')
                lmp.command('pair_coeff * * CH.airebo C')
                lmp.command('minimize 1e-25 1e-25 10000 100000')
                lmp.command('unfix press')
                lmp.command('min_style fire')
                lmp.command('minimize 1e-25 1e-25 1000 10000')

                # get energy for comparison here so we don't neeed to do it in the comparison step
                new_energy = lmp.get_thermo('pe')
                lmp.command('fix fixbonds all bond/create 1 1 1 1.8 1 iparam 3 1 jparam 3 1 atype 1 dtype 1 itype 1')
                lmp.command('run 500')
                lmp.command('unfix fixbonds')
                lmp.command('min_style cg')
                lmp.command('dump 1 all custom 1 smallshapes_periodic_1hex.' + str(k + add) + ' id type x y z c_eng')
                lmp.command(
                    'dump s all local 1 shapebonds.xyz_' + str(k + add) + ' index c_blist[1] c_blist[2] c_blist[3]')
                lmp.command('run 1')
                lmp.command('undump 1')
                lmp.command('undump s')

                # monte-carlo part: accept or reject only if energy of new state is too much higher than
                #   that of old state, *or* if a bond has broken
                # if new_energy > (old_energy + 4.5 + np.exp(-kbT*np.random.rand())) or lmp.gather_bonds()[0] < nbonds:
                if lmp.gather_bonds()[0] < nbonds:
                    # reset ring counts, neighbors, bonding, etc.:
                    rings[r_leave_a1] = rstore[0]
                    rings[r_leave_a2] = rstore[1]
                    rings[r_add_a1] = rstore[2]
                    rings[r_add_a2] = rstore[3]
                    prings[a1] = rstore[4]
                    prings[a2] = rstore[5]
                    bonds[bond] = rstore[6]
                    bonds[b11] = rstore[7]
                    bonds[b12] = rstore[8]
                    bonds[b21] = rstore[9]
                    bonds[b22] = rstore[10]
                    particles[p1] = rstore[11]
                    particles[p2] = rstore[12]
                    particles[p11] = rstore[13]
                    particles[p12] = rstore[14]
                    particles[p21] = rstore[15]
                    particles[p22] = rstore[16]
                    bonds[rstore[17][0]][rstore[17][1]][rstore[17][2]] = p1
                    bonds[rstore[18][0]][rstore[18][1]][rstore[18][2]] = p1
                    bonds[rstore[19][0]][rstore[19][1]][rstore[19][2]] = p2
                    bonds[rstore[20][0]][rstore[20][1]][rstore[20][2]] = p2
                    print('flip rejected, oldE={:.2f}'.format(old_energy), 'newE={:.2f}'.format(new_energy),
                          'nbonds=' + str(lmp.gather_bonds()[0]), file=f)
                    print('Remaining rotations:', unchangedrotsremaining, end=' ', file=f)
                    print('Rings of size', end=' ', file=f)
                    for i in range(maxsize + 1):
                        print(str(i) + ':', str(num[i]) + ',', end=' ', file=f)
                    print('\n', file=f)
                    # then jump back to before rotation
                    lmp.close()
                    lmp = lammps()
                    lmp.command('box tilt large')
                    lmp.command('read_restart restart.rstrt')
                    lmp.command('compute eng all pe/atom')
                    lmp.command('compute eatoms all reduce sum c_eng')
                    lmp.command('compute blist all property/local btype batom1 batom2')
                    lmp.command('pair_style zero 16.0 nocoeff')
                    lmp.command('pair_coeff * * CH.airebo C')
                    lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
                else:
                    num = numi
                    print('flip accepted, oldE={:.2f}'.format(old_energy), 'newE={:.2f}'.format(new_energy),
                          'nbonds=' + str(lmp.gather_bonds()[0]), file=f)
                    print('Remaining rotations:', unchangedrotsremaining, end=' ', file=f)
                    print('Rings of size', end=' ', file=f)
                    for i in range(maxsize + 1):
                        print(str(i) + ':', str(num[i]) + ',', end=' ', file=f)
                    print('\n', file=f)
                    # if not rejected, set new energy of comparison
                    old_energy = new_energy
                    lmp.command('group break1 delete')
                    lmp.command('group break2 delete')
                    lmp.command('group bond1 delete')
                    lmp.command('group bond2 delete')
                    lmp.command('variable dx delete')
                    lmp.command('variable dy delete')
                    lmp.command('variable a1x delete')
                    lmp.command('variable a2x delete')
                    lmp.command('variable a1y delete')
                    lmp.command('variable a2y delete')
                    lmp.command('group set1 delete')
                    lmp.command('group set2 delete')
                    lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
            except Exception as e:
                print('flip rejected:', e, file=f)
                # reset ring counts, neighbors, bonding, etc.:
                rings[r_leave_a1] = rstore[0]
                rings[r_leave_a2] = rstore[1]
                rings[r_add_a1] = rstore[2]
                rings[r_add_a2] = rstore[3]
                prings[a1] = rstore[4]
                prings[a2] = rstore[5]
                bonds[bond] = rstore[6]
                bonds[b11] = rstore[7]
                bonds[b12] = rstore[8]
                bonds[b21] = rstore[9]
                bonds[b22] = rstore[10]
                particles[p1] = rstore[11]
                particles[p2] = rstore[12]
                particles[p11] = rstore[13]
                particles[p12] = rstore[14]
                particles[p21] = rstore[15]
                particles[p22] = rstore[16]
                bonds[rstore[17][0]][rstore[17][1]][rstore[17][2]] = p1
                bonds[rstore[18][0]][rstore[18][1]][rstore[18][2]] = p1
                bonds[rstore[19][0]][rstore[19][1]][rstore[19][2]] = p2
                bonds[rstore[20][0]][rstore[20][1]][rstore[20][2]] = p2

                # then jump back to before rotation
                lmp.close()
                lmp = lammps()
                lmp.command('box tilt large')
                lmp.command('read_restart restart.rstrt')
                lmp.command('compute eng all pe/atom')
                lmp.command('compute eatoms all reduce sum c_eng')
                lmp.command('compute blist all property/local btype batom1 batom2')
                lmp.command('pair_style zero 16.0 nocoeff')
                lmp.command('pair_coeff * * CH.airebo C')
                lmp.command('fix press all box/relax x 0.0 y 0.0 xy 0.0')
        else:
            rings[r_leave_a1] = rstore1[0]
            rings[r_leave_a2] = rstore1[1]
            rings[r_add_a1] = rstore1[2]
            rings[r_add_a2] = rstore1[3]
            prings[a1] = rstore1[4]
            prings[a2] = rstore1[5]
            trynumber = trynumber + 1

for i in range(0, d2):
    x = [len(rings[j + d1 * i]) for j in range(0, d1)]
    for j in range(0, i):
        print('  ', end='', file=f)
    print(x, file=f)

lengths = []
for i in rings:
    lengths.append(len(i))
num = [lengths.count(i) for i in range(maxsize + 1)]
print('Rings of size', end=' ', file=f)
for i in range(maxsize + 1):
    print(str(i) + ':', str(num[i]) + ',', end=' ', file=f)
print('\n', file=f)
quit()

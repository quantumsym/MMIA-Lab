#!~/anaconda3/bin/python


import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams.update({'font.size': 10})


def main(filename, animate=False):

    def animate_func(i):
        plt.pcolormesh(data['lattice_list'][i], cmap='binary')

    data = np.load(filename, 'r')

    print("Length:      %i"%data['length'])
    print("Steps:       %i"%data['iterations'])
    print("Rounds:      %i"%data['rounds'])
    print("Temperature: %f"%data['temperature'])
    print("Ferro_prob:  %f"%data['ferro_prob'])
    #Threads and processes depend on type of run
    try:
        print("Threads:     %i"%data['threads'])
    except KeyError:
        pass
    try:
        print("Processes:   %i"%data['processes'])
    except KeyError:
        pass

    if animate:
        fig = plt.figure()
        plt.hold(True)

        plt.pcolormesh(data['lattice_list'][0], cmap='binary')
        plt.colorbar()

        anim = animation.FuncAnimation(fig, animate_func, frames=range(0,len(data['lattice_list'])-1), blit=False)

        plt.show()
        plt.hold(False)

    else:
        plt.subplot(1, 2, 1)
        plt.pcolormesh(data['lattice_list'][0], cmap='binary')
        plt.colorbar()
        plt.axis('scaled')

        plt.subplot(1, 2, 2)
        plt.pcolormesh(data['lattice_list'][-1], cmap='binary')
        plt.colorbar()
        plt.axis('scaled')

        plt.tight_layout()
        plt.show()

    
    # for lattice in data['lattice_list']:
    #     plt.pcolormesh(lattice, cmap='binary')
    #     plt.colorbar()
    #     plt.pause(0.1)
    #     plt.clf()
    # plt.show()

    # plt.pcolormesh(data['lattice_list'][-1], cmap='binary')
    # plt.colorbar()
    # plt.show()

    x = np.arange(0, data['iterations']+1, data['rounds'])
    x = x/(data['length']**2)

    plt.plot(x, data['energy_list'])
    #plt.xscale('log')
    plt.xlabel(r'Monte Carlo Steps')
    plt.ylabel(r'Energy')
    plt.show()

    plt.plot(x, data['magnetisation_list'])
    #plt.xscale('log')
    plt.xlabel(r'Monte Carlo Steps')
    plt.ylabel(r'm')
    plt.show()

    data.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
            """Plots results from sgi functions.
            """)
    parser.add_argument('filename', type=str, default='results.npz')
    parser.add_argument('--animate', '-a', action='store_true')
    args = parser.parse_args()

    main(args.filename, animate=args.animate)

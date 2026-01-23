import numpy as np
import pyEXP

class AnalyzeMSSA:
    
    def __init__(config, window, npc):
        
        self.mssa = pyEXP.mssa.expMSSA(config_m1_amp, window, npc)
        
    def get_eigenvalues(self):
        
        self.eigenvalues = self.mssa.eigenvalues()
        
        return self.eigenvalues
    
    def plot_eigenvalues(self):
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        ax1.semilogy(self.eigenvalues, '-o')

        ax1.set_xlabel('Index', fontsize=20)
        ax1.set_ylabel('Eigenvalue', fontsize=20)

        ax1.set_title('PC Eigenvalues', fontsize=20)
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/eigenvalues_m1_plot.pdf')
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/eigenvalues_m1_plot.png')
        plt.show()
        
        return fig
    
    def get_PCs(self):
        
        self.pcs = self.mssa.getPC()
        return self.pcs
    
    def plot_PCs(self, coefs):
        
        fig, axs = plt.subplots(7, 1, figsize=(15, 20), sharex=True)

        for i in range(0,4):
            axs[0].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i)) 

        for i in range(4,6):
            axs[1].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i))    

        for i in range(6,8):
            axs[2].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i))    

        for i in range(9,11):
            axs[3].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i))

        for i in range(11,14):
            axs[4].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i))  

        for i in range(14,17):
            axs[5].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i))  

        for i in range(47,50):
            axs[6].plot(coefs.Times()[:nt], self.pcs[:,i], label=str(i))  

        for i in range(7):
            axs[3].set_xlabel('Time', fontsize=20)
            axs[i].set_ylabel('PC Amplitude', fontsize=20)
            axs[i].legend(fontsize=20, loc='lower left')
        plt.suptitle('PC Amplitudes Over Time', fontsize=25)

        fig.tight_layout()
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/pc_plot_m1_amp.pdf')
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/pc_plot_m1_amp.png')
        plt.show()
        
        return fig
    
    def get_power_spectra_DFT(self):
        
        self.freq, self.power = self.mssa.pcDFT()
        return self.freq, self.power
    
    def plot_power_spectra_DFT(self):
        
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(25, 8))

        color = iter(cm.rainbow(np.linspace(0, 1, 11)))

        #for i in range(power.shape[1]):
        for i in range(0, 11,1):
            c = next(color)
            ax1.plot(self.freq[0:30], self.power[0:30,i], '-o', c=c, label=str(i))
            if i>3:
                ax2.plot(self.freq[0:30], self.power[0:30,i], '-o', c=c, label=str(i))

        ax1.set_xlabel('Frequency', fontsize=20)
        ax2.set_xlabel('Frequency', fontsize=20)
        ax1.legend(fontsize=18) ; ax2.legend(fontsize=18)
        ax1.set_ylabel('Power', fontsize=20)

        ax1.set_yscale('log')

        plt.suptitle('Single Armed Amplitude Power Spectra', fontsize=25)
        fig.tight_layout()
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/m1_pc_power_spectra.pdf')
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/m1_pc_power_spectra.png') 
        plt.show()
        
        return fig
    
    def reconstruct(self, n_pcs=50):
        '''
        n_pcs: how many pcs to reconstruct with
        '''
        
        self.mssa.reconstruct([*range(n_pcs)])
        
    def get_wCorr_matrix(self):
        
        self.wCorr = self.mssa.wCorrAll()
        return self.wCorr
    
    def plot_wCorr(self, n_pcs=20):
        '''
        n_pcs: number of pcs to show in the W-Correlation Matrix plot
        '''
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharey=True)
        ax.imshow(self.wCorr[:20, :20], cmap='gray_r')
        #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/wcorr20.png') 
        plt.show()
        
        return fig
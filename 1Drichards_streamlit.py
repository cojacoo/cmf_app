# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

import cmf
from datetime import datetime,timedelta

sns.set_style('whitegrid', {'grid.linestyle': u'--'})

# helper
def plot_rc(retcurve):
    """Plots the retention curve retcurve for the matrix potential values in the array Psi_M
    and the function of K(theta)"""
    Psi_M = np.arange(0, -4, -0.01)

    fig = plt.figure(figsize=(4, 2.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Calculate the wetness at the matrix potential
    W = retcurve.Wetness(Psi_M)
    # plot the retention curve. The methods vgm.Wetness(matrixpotential), vgm.MatricPotential(wetness),
    # vgm.K(wetness) accept numbers as well as arrays for the values
    ax1.plot(Psi_M, W * retcurve.Porosity(0))
    # label axis
    ax1.set_xlabel('Matric potential (m)')
    ax1.set_ylabel(r'water content $\theta (m^3m^{-3})$')

    # Make lower plot (K(W))
    ax2.plot(Psi_M, retcurve.K(W))
    ax2.set_xlabel('Matric potential (m)')
    ax2.set_ylabel(r'$K(\theta) (m day^{-1})$')
    ax2.set_yscale('log')



# Description
st.title('CMF Beispiel für Bodenwasserdynamik')
st.markdown('Dies ist ein Skript für die Modellierung der ungesättigten Wasserbewegung in einer 1-dimensionalen Bodensäule.\nWir verwenden dazu das Paket *CMF*  von Philipp Kraft et al. (2011) https://philippkraft.github.io/cmf, mit dem wir eine 1m tiefe Bodensäule mit homogenem Bodeneigenschaften aufsetzen. Mit dem Modell werden wir die Einstellung des Gleichgewichtszustands der Bodensäule unter unterschiedlichen initialen Bedingungen und mit verschiedenen Parametern der Retentionsbeziehung der Bodenmatrix untersuchen.')

st.sidebar.title('Einstellungen')
det = st.sidebar.checkbox('erweiterte Einstellungen',value=False)
lay2 = False
if det:
    st.sidebar.subheader('Geometry of domain')
    lz = st.sidebar.slider('number of cells', 1, 100, 10)
    dz = st.sidebar.slider('height of each cell (m)', 0.001, 0.5, 0.1)
    st.sidebar.subheader('Soil definition')
    ksat1e = st.sidebar.slider('saturated hydraulic conductivity exponent (m/s)', -8., -2., -6.)
    ksat1 = (10**ksat1e)*(24.*3600.)
    alpha1 = st.sidebar.slider('van Genuchten alpha (1/m)', 0.01, 1.2, 0.035)
    n1 = st.sidebar.slider('van Genuchten n (-)', 1., 4., 1.5)
    
    # Definition einer Retentionsbeziehung des Bodens über die Parameter der van Genuchten-Mualem Funktion
    r_curve = cmf.VanGenuchtenMualem(Ksat=ksat1,phi=0.5,alpha=alpha1,n=n1) #ksat in m/day >> 1.157e-05 m/s
    st.sidebar.pyplot(plot_rc(r_curve))

    lay2 = st.sidebar.checkbox('2 soil layers',value=False)
    if lay2:
        stepz = st.sidebar.slider('first cell of second soil type', 1, lz, lz)
        ksat2e = st.sidebar.slider('2nd saturated hydraulic conductivity exponent (m/s)', -8., -2., -6.)
        ksat2=10**ksat2e*(24.*3600.)
        alpha2 = st.sidebar.slider('2nd van Genuchten alpha (1/m)', 0.01, 1.2, 0.035)
        n2 = st.sidebar.slider('2nd van Genuchten n (-)', 1., 4., 1.5)
        r_curve2 = cmf.VanGenuchtenMualem(Ksat=ksat2,phi=0.5,alpha=alpha2,n=n2) #ksat in m/day >> 1.157e-05 m/s
        st.sidebar.pyplot(plot_rc(r_curve2))

else:
    lz=10
    dz=0.1
    ksat1=5e-6*(24.*3600.)
    alpha1=0.035
    n1=1.5
    r_curve = cmf.VanGenuchtenMualem(Ksat=ksat1,phi=0.5,alpha=alpha1,n=n1) #ksat in m/day >> 1.157e-05 m/s

# Model settings
# Definition eines CMF Projekts
project = cmf.project()

# Nun definieren wir eine Retentionsfunktion für den Boden und fügen unserer Zelle die Bodenschichten mit jeweils 10 cm Mächtigkeit hinzu:
# Definition einer Zelle an Position (0,0,0) mit einer Fläche von 1000 m2 und einem Wasserspeicher an der Oberfläche 
cell = project.NewCell(x=0,y=0,z=0,area=1000, with_surfacewater=True)



# Hinzufügen von lz Schichten mit je dz Mächtigkeit und der oben definierten Retentionsfunktion
for i in range(lz):
    depth = (i+1) * dz
    if lay2:
        if i<stepz:
            cell.add_layer(depth,r_curve)
        else:
            cell.add_layer(depth,r_curve2)
    else:
        cell.add_layer(depth,r_curve)


# Die Zellen werden nun mit der Richards-Gleichung miteinander verbunden:
cell.install_connection(cmf.Richards)

# Und weil es eine Vielzahl von numerischen Verfahren gibt, müssen wir noch den Löser zur Berechnung der internen Flüsse bzw. des Gleichungssystems der Richardsgleichung definieren:
# Definition des Lösers:
#solver = cmf.CVodeKLU(project,1e-6)
#solver = cmf.RKFIntegrator(project,1e-6)
solver = cmf.CVodeIntegrator(project,1e-6)
solver.t = cmf.Time(1,1,2011)


# Nun haben wir eine Bodensäule, deren 10 Schichten mit der Richards-Gleichung in Verbindung stehen. Wie wir in der Vorlesung gelernt haben, kann nun nur etwas passieren, wenn es auch einen Gradienten im totalen hydraulischen Potenzial gibt. Diesen erzeugen wir in unserem Beispiel über die Anfangszustände.
st.sidebar.subheader('Inital conditions and boundary conditions')
cellini = st.sidebar.slider('initial unsaturated state', -20., 0., -5.)
gwpot = st.sidebar.slider('ground water potential', -10., 0., -1.)
surfwat = st.sidebar.slider('surface water depth', 0., 0.5, 0.)

# Dazu sehen wir uns einfach einmal an, wie die Bodensäule als unser Schwamm mit Kontakt zu einer Wasserfläche reagiert:
# Definition eines Grundwasseranschlusses als Randbedingung
gw = project.NewOutlet('groundwater',x=0,y=0,z=-1)
# Definition der Grundwasserhöhe
gw.potential = gwpot
gw.is_source=True
# Verbindung der untersten Zelle mit dem Grundwasser über die Richards-Gleichung
gw_flux=cmf.Richards(cell.layers[-1],gw)

# Definition der Anfangszustände im (ungeättigtem) Boden
# Alle Bodenschichten sind ungesättigt mit einem Potenzial von -5 m
cell.saturated_depth = -1.*cellini

# Definition des Anfangszustandes an der Oberfläche
# 0 mm Wasser zur Infiltration
cell.surfacewater.depth = surfwat


# RUN MODEL
#@st.cache
def runcmf():
    # start with initial conditions
    potential = [cell.layers.potential]
    moisture = [cell.layers.theta]
    tstep = 1./60. #hour
    # The run time loop:
    for t in solver.run(solver.t,
                    solver.t + timedelta(days=60),
                    timedelta(hours=tstep)):
        potential.append(cell.layers.potential)
        moisture.append(cell.layers.theta)

    a = np.arange(lz)
    b = np.repeat('theta',lz)
    c = np.repeat('psi',lz)
    results = pd.concat([pd.Series(np.arange(len(moisture))*tstep),pd.DataFrame(np.array(moisture)),pd.DataFrame(np.array(potential))],axis=1)
    results.columns = ['time_h']+[m+str(n) for m,n in zip(b,a)]+[m+str(n) for m,n in zip(c,a)]
    return results


def plot_results(results,ti):
    a = np.arange(lz)
    b = np.repeat('theta',lz)
    c = np.repeat('psi',lz)
    
    tix = np.where(results.time_h>=ti)[0][0]

    fig = plt.figure(figsize=(10,4),constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = ax1.twiny()
    ax1.plot(results.loc[tix,[m+str(n) for m,n in zip(b,a)]],np.arange(lz)*-1.*dz)
    ax2.plot(results.loc[tix,[m+str(n) for m,n in zip(c,a)]],np.arange(lz)*-1.*dz,'r-')
    ax1.set_title('t='+str(ti)+'h')
    ax1.set_xlim(0.01,0.6)
    ax2.set_xlim(-5.2,-0.8)
    ax2.grid(False)
    
    ax1.set_xlabel(r'Bodenfeuchte $\theta (m^3m^{-3})$')
    ax2.set_xlabel(r'Hydr. Potenzial $\Psi_{tot} (m)$')
    ax1.set_ylabel(r'Tiefe $(m)$')
    
    ax3 = fig.add_subplot(gs[0, 1:])
    ax3.plot(results['time_h'],results[[m+str(n) for m,n in zip(b,a)]])
    ax3.plot([results.loc[tix,'time_h'],results.loc[tix,'time_h']],[0.08,0.52],':',c='gray')
    ax3.set_ylabel(r'Bodenfeuchte $\theta (m^3m^{-3})$')
    
    ax4 = fig.add_subplot(gs[1, 1:])
    for i in np.arange(lz):
        ax4.plot(results['time_h'],results['psi'+str(i)],label=str(np.round(i*dz+dz/2,1))+' m')

    ax4.plot([0,results['time_h'].iloc[-1]],[0,0],'k:',label=r'$\Psi_{tot}$=0')
    ax4.plot([0,results['time_h'].iloc[-1]],[gw.potential,gw.potential],'r:',label=r'$\Psi_{tot}$=equil.')
    ax4.plot([results.loc[tix,'time_h'],results.loc[tix,'time_h']],[-5.2,-0.8],':',c='gray')
    ax4.set_ylabel(r'Hydr. Potenzial $\Psi_{tot} (m)$')
    ax4.set_xlabel(r'$Zeit (h)$')
    ax4.legend(loc=4,ncol=3)
    
    ax3.set_ylim(0.01,0.6)
    #ax4.set_ylim(-5.2,-0.8)

    return fig

def plot_results_im(results,ti):
    a = np.arange(lz)
    b = np.repeat('theta',lz)
    c = np.repeat('psi',lz)
    
    tix = np.where(results.time_h>=ti)[0][0]

    fig = plt.figure(figsize=(10,4),constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = ax1.twiny()
    ax1.plot(results.loc[tix,[m+str(n) for m,n in zip(b,a)]],np.arange(lz)*-1.*dz)
    ax2.plot(results.loc[tix,[m+str(n) for m,n in zip(c,a)]],np.arange(lz)*-1.*dz,'r-')
    ax1.set_title('t='+str(ti)+'h')
    ax1.set_xlim(0.01,0.6)
    ax2.set_xlim(-5.2,-0.8)
    ax2.grid(False)
    
    ax1.set_xlabel(r'Bodenfeuchte $\theta (m^3m^{-3})$')
    ax2.set_xlabel(r'Hydr. Potenzial $\Psi_{tot} (m)$')
    ax1.set_ylabel(r'Tiefe $(m)$')
    
    ax3 = fig.add_subplot(gs[0, 1:])
    plt1 = ax3.imshow(results[[m+str(n) for m,n in zip(b,a)]].values.T,aspect=2300,cmap='Blues',vmin=0.09,vmax=0.6)
    ax3.set_ylabel('Bodenschicht')
    fig.colorbar(plt1, ax = ax3, label='$\theta (m^3m^{-3})$')
    ax3.set_title('Entwicklung der Bodenfeuchte')
    ax3.set_aspect('auto')

    ax4 = fig.add_subplot(gs[1, 1:])    
    plt2 = ax4.imshow(results[[m+str(n) for m,n in zip(c,a)]].values.T,aspect=2300,cmap='viridis_r')
    ax4.set_xlabel('Zeit (min)')
    ax4.set_ylabel('Bodenschicht')
    fig.colorbar(plt2, ax = ax4, label='$\psi_{tot}$ (m)')
    #ax4.plot([results.loc[tix,'time_h']*60.,results.loc[tix,'time_h']*60.],[0,ly],':',c='gray')
    ax4.set_title('Entwicklung des totalen hydraulischen Potenzials')
    ax4.set_aspect('auto')

    return fig

results = runcmf()
implt = st.checkbox('Image Plot',value=False)
ti = st.slider('time of experiment (h)', 0, 60*24, 100)
if implt:
    st.pyplot(plot_results_im(results,ti))
else:
    st.pyplot(plot_results(results,ti))


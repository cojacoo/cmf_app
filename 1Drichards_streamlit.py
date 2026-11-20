# -*- coding: utf-8 -*-

import io
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cmf
from datetime import timedelta

sns.set_style('whitegrid', {'grid.linestyle': u'--'})
st.set_page_config(page_title='Richards 1D', layout='wide')

DAYS = 60  # Simulationsdauer
PHI = 0.5  # Porosität (fix)


def vgm(ksat_ms, alpha, n):
    """van Genuchten-Mualem Retentionsbeziehung, Ksat in m/s -> CMF erwartet m/Tag"""
    return cmf.VanGenuchtenMualem(Ksat=ksat_ms * 86400., phi=PHI, alpha=alpha, n=n)


def plot_rc(retcurve):
    """Retentionskurve theta(psi_m) und Leitfähigkeitsfunktion K(psi_m)"""
    psi_m = np.arange(0, -4, -0.01)
    W = retcurve.Wetness(psi_m)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2.5), constrained_layout=True)
    ax1.plot(psi_m, W * retcurve.Porosity(0))
    ax1.set_xlabel(r'$\psi_m$ (m)')
    ax1.set_ylabel(r'$\theta\ (m^3m^{-3})$')
    ax2.plot(psi_m, retcurve.K(W))
    ax2.set_xlabel(r'$\psi_m$ (m)')
    ax2.set_ylabel(r'$K(\theta)\ (m\ Tag^{-1})$')
    ax2.set_yscale('log')
    return fig


def soil_inputs(suffix, ke_default=-5.3, alpha_default=0.035, n_default=1.5):
    ke = st.sidebar.slider(f'log10 Ksat{suffix} (m/s)', -8., -2., ke_default,
                           help='Gesättigte hydraulische Leitfähigkeit als Zehnerexponent, z.B. -5 = 1e-5 m/s.')
    alpha = st.sidebar.slider(f'van Genuchten α{suffix} (1/m)', 0.01, 1.2, alpha_default,
                              help='≈ Kehrwert des Lufteintrittspunkts. Groß: grober Boden (Sand), klein: feiner Boden (Ton).')
    n = st.sidebar.slider(f'van Genuchten n{suffix} (-)', 1.05, 4., n_default,
                          help='Breite der Porengrößenverteilung. Groß: enge Verteilung, steile Retentionskurve (Sand).')
    rc = vgm(10**ke, alpha, n)
    st.sidebar.pyplot(plot_rc(rc))
    return 10**ke, alpha, n


# ---------------------------------------------------------------- Beschreibung
st.title('CMF Beispiel für Bodenwasserdynamik')
st.markdown('''
Dies ist ein Skript für die Modellierung der ungesättigten Wasserbewegung in einer 1-dimensionalen Bodensäule
mit der **Richards-Gleichung**. Wir verwenden dazu das Paket *CMF* von Philipp Kraft et al. (2011)
https://philippkraft.github.io/cmf, mit dem wir eine (standardmäßig 1 m tiefe) Bodensäule mit homogenen
Bodeneigenschaften aufsetzen. Wir untersuchen, wie sich die Säule unter unterschiedlichen Anfangs- und
Randbedingungen und mit verschiedenen Parametern der Retentionsbeziehung ins **Gleichgewicht** bewegt.
''')

with st.expander('Anleitung: Was wird hier berechnet und was kann ich ausprobieren?'):
    st.markdown(r'''
**Modell.** Die Säule besteht aus übereinanderliegenden Zellen. Zwischen den Zellen fließt Wasser nach Darcy-Buckingham
$q = -K(\theta)\,\partial\Psi_{tot}/\partial z$, mit dem totalen hydraulischen Potenzial
$\Psi_{tot} = \psi_m + z$ (Matrixpotenzial + Gravitationspotenzial, Bezugsniveau $z=0$ an der Bodenoberfläche).
Wasser fließt nur, solange ein Gradient in $\Psi_{tot}$ besteht.

**Randbedingungen.** Unten ist die Säule an ein Grundwasser mit festem Potenzial angeschlossen
(Dirichlet-Randbedingung; es kann Wasser liefern und aufnehmen). Oben kann zu Beginn eine Wasserschicht stehen, die infiltriert.

**Anfangsbedingung.** Die Säule startet im hydrostatischen Gleichgewicht mit einem gedachten Grundwasserspiegel in der
Tiefe des gewählten Anfangspotenzials: $\Psi_{tot}$ ist überall gleich, $\psi_m$ nimmt nach oben ab.
Weicht das Grundwasserpotenzial davon ab, entsteht ein Gradient und Wasser fließt.

**Gleichgewicht.** Am Ende gilt überall $\Psi_{tot} = \Psi_{GW}$ (rote gepunktete Linie). Die Bodenfeuchte ist dann
*nicht* konstant: sie folgt der Retentionskurve, $\theta(z) = \theta(\psi_m = \Psi_{GW} - z)$.

**Bedienung.** Links Anfangs-/Randbedingungen und Bodenparameter einstellen, dann **Modell rechnen** drücken. Rechts die Darstellung wählen (ändert nur die Grafik, kein Neurechnen).

**Aufgaben / Fragen**
1. Standardeinstellung: Wie lange dauert es, bis sich das Gleichgewicht einstellt? Welche Schicht reagiert zuerst?
2. Wie sieht das Feuchteprofil im Gleichgewicht aus und warum ist es nicht homogen?
3. Senken Sie das Grundwasserpotenzial unter das Anfangspotenzial: Was passiert jetzt (Drainage statt Aufsättigung)? Ist der Prozess gleich schnell?
4. Setzen Sie eine Überstauhöhe an der Oberfläche: Wie bewegt sich die Infiltrationsfront?
5. (Erweitert) Vergleichen Sie einen sandigen (großes α, großes n, hohes Ksat) mit einem lehmigen Boden. Wie ändern sich Geschwindigkeit und Gleichgewichtsprofil?
6. (Erweitert) Zwei Schichten: Was passiert an der Schichtgrenze mit $\theta$ und mit $\Psi_{tot}$?
''')

# ---------------------------------------------------------------- Einstellungen
st.sidebar.title('Einstellungen')
runbox = st.sidebar.container()  # Rechenknopf oben, wird unten befüllt
st.sidebar.subheader('Anfangs- und Randbedingungen')
cellini = st.sidebar.slider('Anfangspotenzial Ψtot (m)', -20., 0., -5.,
                            help='Hydrostatische Anfangsbedingung: entspricht einem Grundwasserspiegel in dieser Tiefe unter Flur.')
gwpot = st.sidebar.slider('Grundwasserpotenzial (untere Randbedingung, m)', -10., 0., -1.,
                          help='Festes totales Potenzial des Grundwassers unter der Säule. 0 = Grundwasser bis zur Oberfläche.')
surfwat = st.sidebar.slider('Überstau an der Oberfläche (m)', 0., 0.5, 0.,
                            help='Wasserschicht, die zu Beginn auf der Oberfläche steht und infiltrieren kann.')

det = st.sidebar.checkbox('erweiterte Einstellungen', value=False)
lz, dz = 10, 0.1
soil1 = (5e-6, 0.035, 1.5)
soil2, stepz = None, lz
if det:
    st.sidebar.subheader('Geometrie')
    lz = st.sidebar.slider('Anzahl Zellen', 2, 100, 10)
    dz = st.sidebar.slider('Mächtigkeit je Zelle (m)', 0.01, 0.5, 0.1)
    st.sidebar.caption(f'Säulentiefe: {lz*dz:.2f} m')
    st.sidebar.subheader('Boden')
    soil1 = soil_inputs('')
    if st.sidebar.checkbox('2 Bodenschichten', value=False):
        stepz = st.sidebar.slider('erste Zelle des zweiten Bodens', 2, lz, max(2, lz // 2))
        soil2 = soil_inputs(' (2)', ke_default=-7.)
        st.sidebar.caption(f'Schichtgrenze in {(stepz-1)*dz:.2f} m Tiefe')


# ---------------------------------------------------------------- Modell
# Alle Parameter gehen als Argumente ein: st.cache_data rechnet nur neu, wenn sich ein Argument ändert.
@st.cache_data(show_spinner='Rechne Richards-Gleichung ...')
def runcmf(lz, dz, soil1, soil2, stepz, cellini, gwpot, surfwat):
    project = cmf.project()
    # Zelle mit 1000 m² Fläche und Oberflächenwasserspeicher
    cell = project.NewCell(x=0, y=0, z=0, area=1000, with_surfacewater=True)
    r1 = vgm(*soil1)
    r2 = vgm(*soil2) if soil2 else r1
    for i in range(lz):
        cell.add_layer((i + 1) * dz, r1 if i < stepz - 1 else r2)
    # Schichten untereinander mit der Richards-Gleichung verbinden
    cell.install_connection(cmf.Richards)

    # Grundwasser als untere Randbedingung direkt unter der Säule
    gw = project.NewOutlet('groundwater', x=0, y=0, z=-lz * dz)
    gw.potential = gwpot
    gw.is_source = True
    cmf.Richards(cell.layers[-1], gw)

    # Anfangszustände
    cell.saturated_depth = -cellini
    cell.surfacewater.depth = surfwat

    solver = cmf.RKFIntegrator(project, 1e-6)
    solver.t = cmf.Time(1, 1, 2011)
    tstep = 1. / 60.  # Ausgabeschritt in h (1 min)
    potential = [cell.layers.potential]
    moisture = [cell.layers.theta]
    for _ in solver.run(solver.t, solver.t + timedelta(days=DAYS), timedelta(hours=tstep)):
        potential.append(cell.layers.potential)
        moisture.append(cell.layers.theta)

    cols = [f'theta{i}' for i in range(lz)] + [f'psi{i}' for i in range(lz)]
    results = pd.DataFrame(np.hstack([moisture, potential]), columns=cols)
    results.insert(0, 'time_h', np.arange(len(results)) * tstep)
    return results


def plot_profile(ax1, results, tix, ti, depth, psicols, thetacols, psilim, gwpot):
    ax2 = ax1.twiny()
    ax1.plot(results.loc[tix, thetacols], depth, 'b-')
    ax2.plot(results.loc[tix, psicols], depth, 'r-')
    ax2.axvline(gwpot, c='r', ls=':')
    ax1.set_title(f't = {ti} h', pad=40)
    ax1.set_xlim(0.01, 0.6)
    ax2.set_xlim(*psilim)
    ax2.grid(False)
    ax1.set_xlabel(r'Bodenfeuchte $\theta\ (m^3m^{-3})$', color='b')
    ax2.set_xlabel(r'Hydr. Potenzial $\Psi_{tot}$ (m)', color='r')
    ax1.set_ylabel('Tiefe (m)')


def plot_results(results, ti, image, lz, dz, gwpot):
    thetacols = [f'theta{i}' for i in range(lz)]
    psicols = [f'psi{i}' for i in range(lz)]
    depth = -(np.arange(lz) + 0.5) * dz  # Zellmitten
    tix = np.where(results.time_h >= ti)[0][0]
    t_now = results.loc[tix, 'time_h']
    tmax = results.time_h.iloc[-1]
    psi = results[psicols].values
    psilim = (min(psi.min(), gwpot) - 0.2, max(psi.max(), gwpot) + 0.2)

    fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    plot_profile(fig.add_subplot(gs[:, 0]), results, tix, ti, depth, psicols, thetacols, psilim, gwpot)
    ax3 = fig.add_subplot(gs[0, 1:])
    ax4 = fig.add_subplot(gs[1, 1:], sharex=ax3)

    if image:
        ext = [0, tmax, -lz * dz, 0]
        im1 = ax3.imshow(results[thetacols].values.T, extent=ext, cmap='Blues', vmin=0.01, vmax=0.6, aspect='auto')
        fig.colorbar(im1, ax=ax3, label=r'$\theta\ (m^3m^{-3})$')
        im2 = ax4.imshow(psi.T, extent=ext, cmap='viridis_r', aspect='auto')
        fig.colorbar(im2, ax=ax4, label=r'$\Psi_{tot}$ (m)')
        ax3.set_ylabel('Tiefe (m)')
        ax4.set_ylabel('Tiefe (m)')
    else:
        ax3.plot(results.time_h, results[thetacols])
        ax3.set_ylim(0.01, 0.6)
        ax3.set_ylabel(r'Bodenfeuchte $\theta\ (m^3m^{-3})$')
        for i, c in enumerate(psicols):
            ax4.plot(results.time_h, results[c], label=f'{depth[i]:.2f} m')
        ax4.axhline(gwpot, c='r', ls=':', label=r'$\Psi_{GW}$ = Gleichgewicht')
        ax4.set_ylabel(r'$\Psi_{tot}$ (m)')
        if lz <= 20:
            ax4.legend(loc=4, ncol=3, fontsize='small')
    ax3.set_title('Bodenfeuchte')
    ax4.set_title('Totales hydraulisches Potenzial')
    ax4.set_xlabel('Zeit (h)')
    for ax in (ax3, ax4):
        ax.axvline(t_now, c='gray', ls=':')
    return fig


# Grafik als PNG cachen: Parameteränderungen ohne Neurechnen lösen kein sichtbares Neuzeichnen aus
@st.cache_data(max_entries=200, show_spinner=False)
def render_plot(run, ti, image):
    fig = plot_results(runcmf(**run), ti, image, run['lz'], run['dz'], run['gwpot'])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return buf.getvalue()


# Modell nur auf Knopfdruck rechnen (spart Rechenzeit auf dem Server)
params = dict(lz=lz, dz=dz, soil1=soil1, soil2=soil2, stepz=stepz, cellini=cellini, gwpot=gwpot, surfwat=surfwat)
if runbox.button('Modell rechnen', type='primary', use_container_width=True) or 'params' not in st.session_state:
    st.session_state.params = params
run = st.session_state.params
if run != params:
    runbox.warning('Parameter geändert: bitte „Modell rechnen“ drücken.')
runcmf(**run)  # rechnet nur bei neuen Parametern, sonst Cache

plotcol, ctrlcol = st.columns([4, 1])
with ctrlcol:
    st.subheader('Darstellung')
    implt = st.toggle('Bild (Zeit × Tiefe)', value=False)
    ti = st.slider('Zeitpunkt für das Profil (h)', 0, DAYS * 24, 100,
                   help='Wählt den Zeitpunkt für das Tiefenprofil links (graue Linie rechts).')
    st.caption(f'Gezeigter Lauf: Anfangspotenzial {run["cellini"]} m, Grundwasser {run["gwpot"]} m, '
               f'Überstau {run["surfwat"]} m, {run["lz"]} × {run["dz"]} m Zellen.')
with plotcol:
    st.image(render_plot(run, ti, implt), use_container_width=True)
    st.caption('Links: Profil zum gewählten Zeitpunkt. Rot gepunktet: Grundwasserpotenzial = Gleichgewichtszustand von $\\Psi_{tot}$.')

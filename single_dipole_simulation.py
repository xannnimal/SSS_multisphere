# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:51:07 2023

@author: xanmc
create raw data for single dipole source
from MNE "simulate_evoked_data.py", and my "SSS_with_sphere_opt.py"
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import find_events, Epochs, compute_covariance, make_ad_hoc_cov
from mne.datasets import sample
from mne.simulation import (
    simulate_sparse_stc,
    simulate_raw,
    add_noise,
    add_ecg,
    add_eog,
)
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked
import os
import seaborn as sns
import pandas as pd
import nibabel as nib
import scipy
from scipy.spatial import KDTree
import vedo
from mne.preprocessing import find_bad_channels_maxwell

#print(__doc__)

mindist = 2e-3

# Load real data as templates
data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw = mne.io.read_raw_fif(meg_path / "sample_audvis_raw.fif")
raw.del_proj()  # Get rid of them!
raw.info['bads'] = []
# proj = mne.read_proj(meg_path / "sample_audvis_ecg-proj.fif")
#raw.add_proj(proj)
# raw.info["bads"] = ["MEG 2443", "EEG 053"]  # mark bad channels

######## from SSS sphere opt py
subject = 'sample'
subjects_dir = data_path / 'subjects'
fname_bem = subjects_dir / subject / 'bem' / f'{subject}-5120-5120-5120-bem.fif'
trans = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'
########### back to "simulate_evoked_data.py"

fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
ave_fname = meg_path / "sample_audvis-no-filter-ave.fif"
cov_fname = meg_path / "sample_audvis-cov.fif"

fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info["bads"])
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(ave_fname)

label_names = ["Aud-lh", "Aud-rh"]
labels = [mne.read_label(meg_path / "labels" / f"{ln}.label") for ln in label_names]

# Generate source time courses from 1 dipoles and the corresponding evoked data

times = np.arange(300, dtype=np.float64) / raw.info["sfreq"] - 0.1
rng = np.random.RandomState(42)


def data_fun(times):
    """Generate random source time courses."""
    return (
        50e-9
        * np.sin(30.0 * times)
        * np.exp(-((times - 0.15 + 0.05 * rng.randn(1)) ** 2) / 0.01)
    )


stc = simulate_sparse_stc(
    fwd["src"],
    n_dipoles=1,
    times=times,
    random_state=42,
    labels=labels,
    data_fun=data_fun,
)
# Generate noisy evoked data from "simulate_evoked"
picks = mne.pick_types(raw.info, meg=True, exclude="bads")
iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=180)[1]
nave = 100  # simulate average of 100 epochs

#########changed cov=None to get rid of projection error in SSS calculation
evoked = simulate_evoked(
    fwd, stc, info, cov, nave=nave, use_cps=True, iir_filter=None
)
evoked.plot_white(cov, time_unit="s")

##################################
evoked_meg = evoked.copy().pick_types(meg=True, eeg=False)

raw.crop(tmax=60)
dev_head_t_matrix= raw.info['dev_head_t']

################# from sphereopt.py
bem_surf = mne.read_bem_surfaces(fname_bem) # Boundary Element Method: returns a list of dictionaries
assert bem_surf[0]['id'] == mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD
assert bem_surf[2]['id'] == mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_BRAIN
scalp, _, inner_skull = bem_surf
inside_scalp = mne.surface._CheckInside(scalp, mode='pyvista')
inside_skull = mne.surface._CheckInside(inner_skull, mode='pyvista')
m3_to_cc = 100 ** 3
assert inside_scalp(inner_skull['rr']).all()
assert not inside_skull(scalp['rr']).any()
b = vedo.Mesh([inner_skull['rr'], inner_skull['tris']])
s = vedo.Mesh([scalp['rr'], scalp['tris']])
s_tree = KDTree(scalp['rr'])
brain_volume = b.volume()
print(f'Brain vedo:     {brain_volume * m3_to_cc:8.2f} cc')
brain_vol = nib.load(subjects_dir / subject / 'mri' / 'brainmask.mgz')
brain_rr = np.array(np.where(brain_vol.get_fdata())).T
brain_rr = mne.transforms.apply_trans(brain_vol.header.get_vox2ras_tkr(), brain_rr) / 1000. #apply a transformation matrix
del brain_vol #delete brain volume
brain_rr = brain_rr[inside_skull(brain_rr)]
vox_to_m3 = 1e-9
brain_volume_vox = len(brain_rr) * vox_to_m3

def _print_q(title, got, want):
    title = f'{title}:'.ljust(15)
    print(f'{title} {got * m3_to_cc:8.2f} cc ({(want - got) / want * 100:6.2f} %)')

_print_q('Brain vox', brain_volume_vox, brain_volume_vox)

# 1. Compute a naive sphere using the center of mass of brain surf verts
naive_c = np.mean(inner_skull['rr'], axis=0)
naive_r = np.min(np.linalg.norm(inner_skull['rr'] - naive_c, axis=1))
naive_v = 4 / 3 * np.pi * naive_r ** 3
_print_q('Naive sphere', naive_v, brain_volume)
s1 = vedo.Sphere(naive_c, naive_r, res=100)
_print_q('Naive vedo', s1.volume(), brain_volume)

# 2. Now use the larger radius (to head) plus mesh arithmetic
better_r = s_tree.query(naive_c)[0] - mindist
s1 = vedo.Sphere(naive_c, better_r, res=24)
_print_q('Better vedo', s1.boolean("intersect", b).volume(), brain_volume)
v = np.sum(np.linalg.norm(brain_rr - naive_c, axis=1) <= better_r) * vox_to_m3
_print_q('Better vox', v, brain_volume_vox)

# 3. Now optimize one sphere
from scipy.optimize import fmin_cobyla #constrained optimization by linear approximation

def _cost(c):
    cs = c.reshape(-1, 3)
    rs = np.maximum(s_tree.query(cs)[0] - mindist, 0.)
    resid = brain_volume
    mask = None
    for c, r in zip(cs, rs):
        if not (r and s.is_inside(c)):
            continue
        m = np.linalg.norm(brain_rr - c, axis=1) <= r
        if mask is None:
            mask = m
        else:
            mask |= m
    resid = brain_volume_vox
    if mask is not None:
        resid = resid - np.sum(mask) * vox_to_m3
    #tot = None
    #for c, r in zip(cs, rs):
    #    if not s.is_inside(c):
    #        continue
    #    sph = vedo.Sphere(c, r, res=24)
    #    if tot is None:
    #        tot = sph
    #    elif not tot.is_inside(sph.rr).all():
    #        print(np.linalg.norm(tot.center - c), tot.radius, r)
    #        tot = tot.boolean("+", sph)
    #resid = brain_volume
    #if tot is not None:
    #    resid = resid - tot.boolean("intersect", b).volume()
    # print(f'  {len(cs)} resid:   {resid * m3_to_cc:8.2f} cc ({resid / brain_volume * 100:6.2f} %)')
    return resid

def _cons(c):
    cs = c.reshape(-1, 3)
    sign = np.array([2 * s.is_inside(c) - 1 for c in cs], float)
    cons = sign * s_tree.query(cs)[0] - mindist
    return cons

x = naive_c
c_opt_1 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
v_opt_1 = brain_volume_vox - _cost(c_opt_1)
_print_q('COBYLA 1', v_opt_1, brain_volume_vox)

# 4. Now optimize two spheres
x = np.concatenate([c_opt_1, naive_c])
c_opt_2 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
v_opt_2 = brain_volume_vox - _cost(c_opt_2)
_print_q('COBYLA 2', v_opt_2, brain_volume_vox)

# 4. Finally, three spheres (not perfect, not global opt)
x = np.concatenate([c_opt_2, naive_c])
c_opt_3 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
v_opt_3 = brain_volume_vox - _cost(c_opt_3)
_print_q('COBYLA 3', v_opt_3, brain_volume_vox)

#Left out all of the plotting in sphere_opt.py

# Output 2-sphere case
mri_head_t = mne.transforms.invert_transform(mne.read_trans(trans)) #trans is the raw_fif_trans file
assert mri_head_t['from'] == mne.io.constants.FIFF.FIFFV_COORD_MRI, mri_head_t['from']
for use in (c_opt_1, c_opt_2):
    centers = mne.transforms.apply_trans(mri_head_t, use.reshape(-1, 3))
    print(centers)

two_sphere_center1 = centers[0]
two_sphere_center2 = centers[1]

#################################################33
fine_cal_file = os.path.join(data_path, 'SSS', 'sss_cal_mgh.dat') #site-specific information about sensor orientation and calibration
crosstalk_file = os.path.join(data_path, 'SSS', 'ct_sparse_mgh.fif')
###############
# fig = mne.viz.plot_alignment(
#     raw.info,
#     trans=trans,
#     subject="sample",
#     subjects_dir=subjects_dir,
#     surfaces="head-dense",
#     show_axes=True,
#     dig=True,
#     eeg=[],
#     meg="sensors",
#     coord_frame="meg",
#     mri_fiducials="estimated",
# )
# mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

########### SSS basis and Maxwell Filtering###############
#raw_sss=basis to reconstruct data, pS=stabilized inv of S array, reg_moments=kept moements, n_use_in=num of kept moments
[S_0sphere, pS_0, reg_moments_0, n_use_in_0]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin='auto', int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
[S_1sphere, pS_1, reg_moments_1, n_use_in_1]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=naive_c, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
[S_2sphere_1, pS_2_1, reg_moments_2_1, n_use_in_2_1]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center1, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
[S_2sphere_2, pS_2_2, reg_moments_2_2, n_use_in_2_2]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center2, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)

############Reconstruct Data######################
# reconstruct data using the SSS basis up to the number of moments for the in basis "n_use_in"
#auto Origin
phi_in_0 = S_0sphere[:, :n_use_in_0] @ pS_0[:n_use_in_0] @ evoked_meg._data[:306]
phi_in_t0= np.matrix.transpose(phi_in_0)

#single sphere origin
phi_in_1 = S_1sphere[:, :n_use_in_1] @ pS_1[:n_use_in_1] @ evoked_meg._data[:306]
phi_in_t1= np.matrix.transpose(phi_in_1)

#two sphere origin 1
phi_in_2_1 = S_2sphere_1[:, :n_use_in_2_1] @ pS_2_1[:n_use_in_2_1] @ evoked_meg._data[:306]
phi_in_t2_1= np.matrix.transpose(phi_in_2_1)

#two sphere origin 2
phi_in_2_2 = S_2sphere_2[:, :n_use_in_2_2] @ pS_2_2[:n_use_in_2_2] @ evoked_meg._data[:306]
phi_in_t2_2= np.matrix.transpose(phi_in_2_2)

##########MAKE PLOTS##################
plt.plot(evoked_meg.times,phi_in_0[0,:], label='reconstructed')
plt.plot(evoked_meg.times,evoked_meg.data[0,:], label = 'original')
plt.legend(loc='upper left')
plt.title('Channel 1 Evoked MEG, Auto origin')
plt.show()

#individual plots of raw, auto oringin, and optimized one sphere origin
plt.plot(evoked_meg.times, np.matrix.transpose(evoked_meg.data[:,:]))
plt.title('Evoked MEG Raw Data')
plt.show()

plt.plot(evoked_meg.times,phi_in_t0[:,:])
plt.title('Reconstructed, Auto Origin')
plt.show()

# plt.plot(evoked_meg.times,phi_in_t1[:,:])
# plt.title('Reconstructed, Optimized One Sphere Origin')
# plt.show()

# plt.plot(evoked_meg.times,phi_in_t2_1[:,:])
# plt.title('Reconstructed, Optimized two Sphere Origin one')
# plt.show()

# plt.plot(evoked_meg.times,phi_in_t2_2[:,:])
# plt.title('Reconstructed, Optimized two Sphere Origin two')
# plt.show()

##########################


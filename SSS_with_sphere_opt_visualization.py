# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:49:29 2023

@author: xanmc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:21:45 2023

@author: xanmc

Using one sphere and two sphere origins found from "sphere_opt.py" written by Eric Larson Jan 2023
Use origins for SSS basis explansions modified from "Signal-space separation (SSS) and Maxwell filtering" 
tutorial included in MNE-pythonS. SSS is not included, using code to work on plotting the MEG helmet, head, and basis spheres
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mne
import nibabel as nib
import scipy
import scipy.io
from scipy.spatial import KDTree
import vedo
from mne.preprocessing import find_bad_channels_maxwell

mindist = 2e-3

data_path = mne.datasets.sample.data_path()
subject = 'sample'
subjects_dir = data_path / 'subjects'
fname_bem = subjects_dir / subject / 'bem' / f'{subject}-5120-5120-5120-bem.fif'
trans = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(data_path, 'MEG', 'sample','sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)


#create evoked data
events = mne.find_events(raw, stim_channel='STI 014')
# we'll skip the "face" and "buttonpress" conditions to save memory
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4}
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.2, event_id=event_dict,
                    preload=True, proj=False)
evoked = epochs['auditory/left'].average()
evoked_meg = evoked.copy().pick_types(meg=True, eeg=False)

raw.crop(tmax=60)

#evoked_meg.plot(spatial_colors=True)
#raw.plot(events=None, duration=10)

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

###########################################
#plot the spheres and the brain to see the overlap
import pyvista as pv
import pyvistaqt
plotter = pyvistaqt.BackgroundPlotter(
     shape=(1, 3), window_size=(1200, 300), #(1,3) for three sphere case
     editor=False, menu_bar=False, toolbar=False)
plotter.background_color = 'w'
brain_mesh = pv.helpers.make_tri_mesh(inner_skull['rr'], inner_skull['tris'])
scalp_mesh = pv.helpers.make_tri_mesh(scalp['rr'], scalp['tris'])
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
mesh_kwargs = dict(render=False, reset_camera=False, smooth_shading=True)
for ci, cs in enumerate((c_opt_1, c_opt_2, c_opt_3)):
    plotter.subplot(0, ci)
    plotter.camera.position = (0., -0.5, 0)
    plotter.camera.focal_point = (0., 0., 0.)
    plotter.camera.azimuth = 90
    plotter.camera.elevation = 0
    plotter.camera.up = (0., 0., 1.)
    plotter.add_mesh(brain_mesh, opacity=0.2, color='k', **mesh_kwargs)
    plotter.add_mesh(scalp_mesh, opacity=0.1, color='tan', **mesh_kwargs)
    for c, color in zip(cs.reshape(-1, 3), colors):
        sphere = pv.Sphere(s_tree.query(c)[0] - mindist, c)
        plotter.add_mesh(sphere, opacity=0.5, color=color, **mesh_kwargs)
plotter.show()

# Output 2-sphere case
mri_head_t = mne.transforms.invert_transform(mne.read_trans(trans)) #trans is the raw_fif_trans file
assert mri_head_t['from'] == mne.io.constants.FIFF.FIFFV_COORD_MRI, mri_head_t['from']
for use in (c_opt_1, c_opt_2):
    centers = mne.transforms.apply_trans(mri_head_t, use.reshape(-1, 3))
    print(centers)

two_sphere_center1 = centers[0]
two_sphere_center2 = centers[1]



#################################################
#plot the brain and sensor array with origins and vector orientations
fine_cal_file = os.path.join(data_path, 'SSS', 'sss_cal_mgh.dat') #site-specific information about sensor orientation and calibration
crosstalk_file = os.path.join(data_path, 'SSS', 'ct_sparse_mgh.fif')
###############
fig = mne.viz.plot_alignment(
    raw.info,
    trans=trans,
    subject="sample",
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    show_axes=True,
    dig=True,
    eeg=[],
    meg="sensors",
    coord_frame="meg",
    mri_fiducials="estimated",
)
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))


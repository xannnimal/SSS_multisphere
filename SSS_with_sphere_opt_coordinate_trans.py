# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:21:45 2023

@author: xanmc

COPY to investigate the effects of roatiting the coordinate basis in 'dev_head_t' on the resulting
SSS bases calculations and subspace angles.

Using one sphere and two sphere origins found from "sphere_opt.py" written by Eric Larson Jan 2023
Use origins for SSS basis explansions modified from "Signal-space separation (SSS) and Maxwell filtering" 
tutorial included in MNE-pythonS
"""

import os
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
import matplotlib
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

################# ROTATE 'dev_head_t' matrix########### between head and MEG

#evoked_meg.plot(spatial_colors=True)
#raw.plot(events=None, duration=10)
dev_head_t = raw.info['dev_head_t']
dev_head_t_matrix_4 = raw.info['dev_head_t']['trans']


############### change to identity matrix with same translation vector
new_dev_3x3=np.array(((1,0,0),(0,1,0),(0,0,1)))
dev_head_t_matrix_4[:3,:3]=new_dev_3x3
raw.info['dev_head_t']['trans'] = dev_head_t_matrix_4
check=raw.info['dev_head_t']['trans']

####### make rotation matrix 
angle =5 #change to any angle
theta= np.radians(angle)
c = np.cos(theta)
s= np.sin(theta)
#z axis rotations
R = np.array(((c,-s,0),(s,c,0), (0,0,1)))
#x axis rotations
# R = np.array(((1,0,0),(0,c,-s), (0,s,c)))

#### rotate dev_head_t_matrix_3
#isolate 3x3 part
dev_head_t_matrix_3=dev_head_t_matrix_4[:3,:3]
dev_head_t_rotated_3=np.matmul(R,dev_head_t_matrix_3)
#combine back into 4x4 matrix
dev_head_t_rotated_4 = dev_head_t_matrix_4
dev_head_t_rotated_4[:3,:3] = dev_head_t_rotated_3
#change the raw.info transformation matrix to equal our new rotated one
raw.info['dev_head_t']['trans'] = dev_head_t_rotated_4
check2 = raw.info['dev_head_t']['trans']

#new_dev= np.array(((0.99142,-0.0399364,-0.124467,0),(0.0606612,0.984012,0.167456,0),(0,0,1,0),(0.11579,-0.17357,0.977991,1)))
#new_dev = np.array(((1,0,0,-0.17933271),(0,1,0,2.176516),(0,0,1,-0.0150475),(0,0,0,1)))
#raw.info['dev_head_t']['trans'] = new_dev
#check = raw.info['dev_head_t']['trans']

####################################
############# try Rotate 'trans' between head and MRI instead
mri_head_t = mne.transforms.invert_transform(mne.read_trans(trans)) #trans is the raw_fif_trans file

##############change mri matric to identity 
mri4x4 = mri_head_t['trans']
new_mri_3x3=np.array(((1,0,0),(0,1,0),(0,0,1)))
mri4x4[:3,:3]=new_mri_3x3
mri_head_t['trans']=mri4x4
check3= mri_head_t['trans']

##############try rotating mri_head_t
# angle =90 #change to any angle
# theta= np.radians(angle)
# c = np.cos(theta)
# s= np.sin(theta)
# #z axis rotations
# R = np.array(((c,-s,0),(s,c,0), (0,0,1)))
# #x axis rotations
# # R = np.array(((1,0,0),(0,c,-s), (0,s,c)))
# mri4x4 = mri_head_t['trans']
# mri3x3 = mri4x4[:3,:3]
# rotated_mri3x3=np.matmul(R,mri3x3)
# rotated_mri4x4=mri4x4
# rotated_mri4x4[:3,:3] = rotated_mri3x3
# mri_head_t['trans'] = rotated_mri4x4
# check2= mri_head_t['trans']
# check3 = mne.transforms.invert_transform(mne.read_trans(trans))



#############################################
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


################# from sphereopt.py ###################
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
#centers are found using the MRI transformation matrix.


assert mri_head_t['from'] == mne.io.constants.FIFF.FIFFV_COORD_MRI, mri_head_t['from']
for use in (c_opt_1, c_opt_2):
    centers = mne.transforms.apply_trans(mri_head_t, use.reshape(-1, 3))
    print(centers)

two_sphere_center1 = centers[0]
two_sphere_center2 = centers[1]



################# Plot 3D head+sensor pic ################################
fine_cal_file = os.path.join(data_path, 'SSS', 'sss_cal_mgh.dat') #site-specific information about sensor orientation and calibration
crosstalk_file = os.path.join(data_path, 'SSS', 'ct_sparse_mgh.fif')

fig = mne.viz.plot_alignment(
    raw.info,
    trans=mri_head_t,
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

###########plot spheres
# import pyvista as pv
# import pyvistaqt
# plotter = pyvistaqt.BackgroundPlotter(
#     shape=(1, 3), window_size=(1200, 300),
#     editor=False, menu_bar=False, toolbar=False)
# plotter.background_color = 'w'
# brain_mesh = pv.helpers.make_tri_mesh(inner_skull['rr'], inner_skull['tris'])
# scalp_mesh = pv.helpers.make_tri_mesh(scalp['rr'], scalp['tris'])
# colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
# mesh_kwargs = dict(render=False, reset_camera=False, smooth_shading=True)
# for ci, cs in enumerate((c_opt_1, c_opt_2, c_opt_3)):
#     plotter.subplot(0, ci)
#     plotter.camera.position = (0., -0.5, 0)
#     plotter.camera.focal_point = (0., 0., 0.)
#     plotter.camera.azimuth = 90
#     plotter.camera.elevation = 0
#     plotter.camera.up = (0., 0., 1.)
#     plotter.add_mesh(brain_mesh, opacity=0.2, color='k', **mesh_kwargs)
#     plotter.add_mesh(scalp_mesh, opacity=0.1, color='tan', **mesh_kwargs)
#     for c, color in zip(cs.reshape(-1, 3), colors):
#         sphere = pv.Sphere(s_tree.query(c)[0] - mindist, c)
#         plotter.add_mesh(sphere, opacity=0.5, color=color, **mesh_kwargs)
#plotter.show()




########### SSS basis and Maxwell Filtering###############
#raw_sss=basis to reconstruct data, pS=stabilized inv of S array, reg_moments=kept moements, n_use_in=num of kept moments
[S_0sphere_z5_2, pS_0, reg_moments_0, n_use_in_0]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=[0,0,0], int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
#[SS_02_1sphere, pS_1, reg_moments_1, n_use_in_1]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=naive_c, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
#[S_2sphere_1_z360, pS_2_1_z360, reg_moments_2_1_z360, n_use_in_2_1_z360]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center1, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
#[S_2sphere_2_z360, pS_2_2_z360, reg_moments_2_2_z360, n_use_in_2_2_z360]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center2, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)

# [S_2sphere_1_n90, pS_2_1_n90, reg_moments_2_1_n90, n_use_in_2_1_n90]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center1, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_2sphere_2_n90, pS_2_2_n90, reg_moments_2_2_n90, n_use_in_2_2_n90]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center2, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)


###save S-2sphere 1 and 2 to files to open in matlab
#scipy.io.savemat('C:/Users/xanmc/RESEARCH/rotation_identity_aligned_axes_2/S_0sphere_z5_2.mat', mdict={'S_0sphere_z5_2': S_0sphere_z5_2})


#scipy.io.savemat('C:/Users/xanmc/RESEARCH/rotation_0origin/S_0sphere_z360.mat', mdict={'S_0sphere_z360': S_0sphere_z360})
#scipy.io.savemat('C:/Users/xanmc/RESEARCH/new_rotations_surface_plot/S_2sphere_2_z360.mat', mdict={'S_2sphere_2_z360': S_2sphere_2_z360})

#scipy.io.savemat('C:/Users/xanmc/S_2sphere_1_n360x.mat', mdict={'S_2sphere_1_n360x': S_2sphere_1_n360x})
#scipy.io.savemat('C:/Users/xanmc/S_2sphere_2_n360x.mat', mdict={'S_2sphere_2_n360x': S_2sphere_2_n360x})


############Reconstruct Data######################
# reconstruct data using the SSS basis up to the number of moments for the in basis "n_use_in"
#auto Origin
# phi_in_0 = S_0sphere[:, :n_use_in_0] @ pS_0[:n_use_in_0] @ evoked_meg._data[:306]
# phi_in_t0= np.matrix.transpose(phi_in_0)

# #single sphere origin
# phi_in_1 = S_1sphere[:, :n_use_in_1] @ pS_1[:n_use_in_1] @ evoked_meg._data[:306]
# phi_in_t1= np.matrix.transpose(phi_in_1)

#two sphere origin 1
# phi_in_2_1_n = S_2sphere_1_n[:, :n_use_in_2_1_n] @ pS_2_1_n[:n_use_in_2_1_n] @ evoked_meg._data[:306]
# phi_in_t2_1_n= np.matrix.transpose(phi_in_2_1_n)

# #two sphere origin 2
# phi_in_2_2_n = S_2sphere_2_n[:, :n_use_in_2_2_n] @ pS_2_2_n[:n_use_in_2_2_n] @ evoked_meg._data[:306]
# phi_in_t2_2_n= np.matrix.transpose(phi_in_2_2_n)

#####CALCULATE MANY ORIGINS TO TEST SUBSPACE ANGLE#################
#keep sphere1 center the same, keep x,z coord of sphere 2 the same, change y
#define centers

# center_015 = np.array([two_sphere_center1[0], 0.015, two_sphere_center1[2]])
# center_02 = np.array([two_sphere_center1[0], 0.02, two_sphere_center1[2]])
# center_025 = np.array([two_sphere_center1[0], 0.025, two_sphere_center1[2]])
# center_03 = np.array([two_sphere_center1[0], 0.03, two_sphere_center1[2]])
# center_035 = np.array([two_sphere_center1[0], 0.035, two_sphere_center1[2]])

# center_z053 = np.array([two_sphere_center1[0], two_sphere_center1[1], 0.053])
# center_z055 = np.array([two_sphere_center1[0], two_sphere_center1[1], 0.055])
# center_z057 = np.array([two_sphere_center1[0], two_sphere_center1[1], 0.057])

# center_y015_z053 = np.array([two_sphere_center1[0], 0.015, 0.053])
# center_y025_z055 = np.array([two_sphere_center1[0], 0.025, 0.055])
# center_y035_z057 = np.array([two_sphere_center1[0], 0.035, 0.057])

# [S_015_1, pS_015, reg_moments_015, n_use_in_015]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_015, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_02_1, pS_02, reg_moments_02, n_use_in_02]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_02, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_025_1, pS_025, reg_moments_025, n_use_in_025]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_025, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_03_1, pS_03, reg_moments_03, n_use_in_03]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_03, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_035_1, pS_035, reg_moments_035, n_use_in_035]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_035, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)

# [S_z053_1, pS_z053, reg_moments_z053, n_use_in_z053]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_z053, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_z055_1, pS_z055, reg_moments_z055, n_use_in_z055]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_z055, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_z057_1, pS_z057, reg_moments_z057, n_use_in_z057]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_z057, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)

# [S_y015_z053, pS_y015_z053, reg_moments_y015_z053, n_use_in_z015_y053]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_y015_z053, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_y025_z055, pS_y025_z055, reg_moments_y025_z055, n_use_in_z025_y055]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_y025_z055, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_y035_z057, pS_y035_z057, reg_moments_y035_z057, n_use_in_z035_y057]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=center_y035_z057, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='ignore', mag_scale=100.0, extended_proj=(), verbose=None)


# scipy.io.savemat('C:/Users/xanmc/S_015_1.mat', mdict={'S_015_1': S_015_1})
# scipy.io.savemat('C:/Users/xanmc/S_02_1.mat', mdict={'S_02_1': S_02_1})
# scipy.io.savemat('C:/Users/xanmc/S_025_1.mat', mdict={'S_025_1': S_025_1})
# scipy.io.savemat('C:/Users/xanmc/S_03_1.mat', mdict={'S_03_1': S_03_1})
# scipy.io.savemat('C:/Users/xanmc/S_035_1.mat', mdict={'S_035_1': S_035_1})

# scipy.io.savemat('C:/Users/xanmc/S_z053_1.mat', mdict={'S_z053_1': S_z053_1})
# scipy.io.savemat('C:/Users/xanmc/S_z055_1.mat', mdict={'S_z055_1': S_z055_1})
# scipy.io.savemat('C:/Users/xanmc/S_z057_1.mat', mdict={'S_z057_1': S_z057_1})

# scipy.io.savemat('C:/Users/xanmc/S_y015_z053_1.mat', mdict={'S_y15_z053': S_y015_z053})
# scipy.io.savemat('C:/Users/xanmc/S_y025_z055_1.mat', mdict={'S_y25_z055': S_y025_z055})
# scipy.io.savemat('C:/Users/xanmc/S_y035_z057_1.mat', mdict={'S_y35_z057': S_y035_z057})




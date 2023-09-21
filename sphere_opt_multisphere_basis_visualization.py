# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:21:45 2023

@author: xanmc

Using one sphere and two sphere origins found from "sphere_opt.py" written by Eric Larson Jan 2023
Use origins for SSS basis explansions modified from "Signal-space separation (SSS) and Maxwell filtering" 
tutorial included in MNE-pythonS
"""

import os
#import vpython
#from vpython import *
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
#sphere_naive = sphere(pos=vector(0.000673388,-0.0100136,0.0442627), radius=naive_r)
#sphere_naive=pyvista.sphere(radius=naive_r, center=naive_c, direction=(0.0, 0.0, 1.0), theta_resolution=30, phi_resolution=30, start_theta=0.0, end_theta=360.0, start_phi=0.0, end_phi=180.0)


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
    #fig=s1,
)
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

# print(
#     "Distance from head origin to MEG origin: %0.1f mm"
#     % (1000 * np.linalg.norm(raw.info["dev_head_t"]["trans"][:3, 3]))
# )
# print(
#     "Distance from head origin to MRI origin: %0.1f mm"
#     % (1000 * np.linalg.norm(trans["trans"][:3, 3]))
# )
# dists = mne.dig_mri_distances(raw.info, trans, "sample", subjects_dir=subjects_dir)
# print(
#     "Distance from %s digitized points to head surface: %0.1f mm"
#     % (len(dists), 1000 * np.mean(dists))
# )


#############################
# ## look for bad channels
# raw.info['bads'] = []
# raw_check = raw.copy()
# auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
#     raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
#     return_scores=True, verbose=True)
# #print(auto_noisy_chs)  # we should find them!
# #print(auto_flat_chs)  # none for this dataset
# bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
# raw.info['bads'] = bads
# raw.info['bads'] += ['MEG 2313']  # from manual inspection

########### SSS basis and Maxwell Filtering###############
#raw_sss=basis to reconstruct data, pS=stabilized inv of S array, reg_moments=kept moements, n_use_in=num of kept moments
# [S_0sphere, pS_0, reg_moments_0, n_use_in_0]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin='auto', int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_1sphere, pS_1, reg_moments_1, n_use_in_1]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=naive_c, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_2sphere_1, pS_2_1, reg_moments_2_1, n_use_in_2_1]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center1, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)
# [S_2sphere_2, pS_2_2, reg_moments_2_2, n_use_in_2_2]=mne.preprocessing.compute_maxwell_basis(evoked_meg.info, origin=two_sphere_center2, int_order=8, ext_order=3, calibration=None, coord_frame='head', regularize=None, ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)

# ###save S-2sphere 1 and 2 to files to open in matlab
# scipy.io.savemat('C:/Users/xanmc/S_2sphere_1.mat', mdict={'S_2sphere_1': S_2sphere_1})
# scipy.io.savemat('C:/Users/xanmc/S_2sphere_2.mat', mdict={'S_2sphere_2': S_2sphere_2})

############Reconstruct Data######################
# reconstruct data using the SSS basis up to the number of moments for the in basis "n_use_in"
#auto Origin
# phi_in_0 = S_0sphere[:, :n_use_in_0] @ pS_0[:n_use_in_0] @ evoked_meg._data[:306]
# phi_in_t0= np.matrix.transpose(phi_in_0)

# #single sphere origin
# phi_in_1 = S_1sphere[:, :n_use_in_1] @ pS_1[:n_use_in_1] @ evoked_meg._data[:306]
# phi_in_t1= np.matrix.transpose(phi_in_1)

# #two sphere origin 1
# phi_in_2_1 = S_2sphere_1[:, :n_use_in_2_1] @ pS_2_1[:n_use_in_2_1] @ evoked_meg._data[:306]
# phi_in_t2_1= np.matrix.transpose(phi_in_2_1)

# #two sphere origin 2
# phi_in_2_2 = S_2sphere_2[:, :n_use_in_2_2] @ pS_2_2[:n_use_in_2_2] @ evoked_meg._data[:306]
# phi_in_t2_2= np.matrix.transpose(phi_in_2_2)

#two-sphere cases might be doing something weird because one of the coordinates is zero?
#how to calculate the Sout basis from this?

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



######### Subspace Angle Calcs #######
#sub_angle = scipy.linalg.subspace_angles(phi_in_2_1, phi_in_2_2)
#function adapted from subspace.m MATLAB
# def my_sub_angles(A,B):
#     threshold = np.sqrt(2)/2
#     trans_A = np.transpose(A)
#     s = np.matmul(trans_A,B)
#     costheta = np.amin(s)
#     #print(costheta)
#     if costheta < threshold:
#         minimum_val = np.minimum(1,costheta)
#         #print(minimum_val)
#         theta = np.arccos(minimum_val) #error with ?
#     else:
#         # if np.ndarray.size(A,2)<np.ndarray.size(B,2):
#         #     mat = trans_A-np.matmul((np.matmul(trans_A,B)),np.transpose(B))
#         #     sintheta = scipy.linalg.norm(mat)
#         #     theta = np.asin(np.minimum(1,sintheta))
#         # else:
#         mat = np.transpose(B)-(np.matmul(np.transpose(B),A))*trans_A
#         sintheta = scipy.linalg.norm(mat)
#         min_val=np.minimum(1,sintheta)
#         theta = np.arcsin(min_val)
#     return theta

#sub_angles_1 = my_sub_angles(S_2sphere_2[:,83], S_2sphere_1[:,83])
#one basis is larger than the other?
#maybe = S_2sphere_2[:,1]
# size = S_2sphere_2.shape[1]
# sub_angles=np.array([])
# for i in range(0,size): #find a way to do the size as # of rows
#     #compare one collumn in S_2sphere_1 to every collumn in  to get subspace angles
#     theta = my_sub_angles(S_2sphere_2[:,i], S_2sphere_1[:,i]) #all the rows in the i'th collumn
#     sub_angles=np.append(sub_angles,theta)
    

##########MAKE PLOTS##################
# plt.plot(evoked_meg.times,phi_in_0[0,:], label='reconstructed')
# plt.plot(evoked_meg.times,evoked_meg.data[0,:], label = 'original')
# plt.legend(loc='upper left')
# plt.title('Channel 1 Evoked MEG, Auto origin')
# plt.show()

#individual plots of raw, auto oringin, and optimized one sphere origin
# plt.plot(evoked_meg.times, np.matrix.transpose(evoked_meg.data[:,:]))
# plt.title('Evoked MEG Raw Data')
# plt.show()

# plt.plot(evoked_meg.times,phi_in_t0[:,:])
# plt.title('Reconstructed, Auto Origin')
# plt.show()

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
#code that I'm not using but I'm afraid to delete forever

#find some well known data, simulate a brain signal with external interference (one dipole)
#check conventionally with one origin, Sin and Sout, with a typical SQUID sensor array
#show you can distinguish between internal signals and ext as we typically do
#merege Sin1 and sin2 and reconstruct, then compare 
#then try with the OPM geometry 


#raw_sss = mne.preprocessing.maxwell_filter(raw, origin=naive_c, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True)
# times = np.linspace(-0.2, 0.6, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

# evoked_meg_times_indicies = evoked_meg.time_as_index(times)
# evoked_meg_1ms = evoked.data[:,:300]
# plt.plot(evoked_meg_1ms)
#raw.pick(['meg']).plot(duration=2, butterfly=True)

#raw_meg = raw.get_data(picks='meg')

#x_hat = pS_1[:, :n_use_in_1] @ raw_meg[:n_use_in_1]
#phi_in_1=S_1sphere[:, :n_use_in_1] @ x_hat
# does this equation actually match Eqs 37 and 38? 
#Should the signal vector phi be raw data, "evoked" data from epoch auditory/left? Data for a specific time frame?
#do you use the full raw data to calculate SSS or make a new basis for each time?
# does the "epoch" time match the time in the raw data file? Seems like it starts with negative t

#raw data compared to auto origin and calculated one-sphere center
# fig, axs = plt.subplots(3)
# fig.tight_layout(pad=2.5)
# axs[0].plot(evoked_meg.times, np.matrix.transpose(evoked_meg.data))
# axs[0].set_title('Raw Data')
# axs[1].plot(evoked_meg.times,phi_in_t0[:,:])
# axs[1].set_title('Reconstructed, Auto Origin')
# axs[2].plot(evoked_meg.times,phi_in_t1[:,:])
# axs[2].set_title('Reconstructed, One Sphere Origin')
# fig.show()

#two sphere centers 1 and 2
# fig, axs = plt.subplots(2)
# fig.tight_layout(pad=2.5)
# axs[0].plot(evoked_meg.times,phi_in_t2_1[:,:])
# axs[0].set_title('Reconstructed, Two Sphere, Origin 1')
# axs[1].plot(evoked_meg.times,phi_in_t2_2[:,:])
# axs[1].set_title('Reconstructed, Two Sphere, Origin 2')
# fig.show()
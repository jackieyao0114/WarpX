####################################################################################################
## This input file is a convergence test for a coupled Maxwell+LLG system                         ##
####################################################################################################

################################
####### GENERAL PARAMETERS ######
#################################
max_step = 5e3
amr.n_cell = 64 64 512 # number of cells spanning the domain in each coordinate direction at level 0
amr.max_grid_size = 32 # maximum size of each AMReX box, used to decompose the domain
amr.blocking_factor = 16
geometry.coord_sys = 0
geometry.is_periodic = 1 1 0

geometry.prob_lo = -15e-3 -15e-3 -30.0e-3
geometry.prob_hi =  15e-3  15e-3  30.0e-3

amr.max_level = 0

my_constants.pi = 3.14159265359
my_constants.L = 2.4e-1
my_constants.h = 2.0e-3 # thickness of the film
my_constants.c = 299792458.
my_constants.wavelength = 0.2308 # frequency is 1.30 GHz
my_constants.TP = 1.5385e-9 # Gaussian pulse width

#################################
############ NUMERICS ###########
#################################
warpx.verbose = 0
warpx.use_filter = 0
warpx.cfl = 0.9
warpx.do_pml_Lo = 0 0 0
warpx.do_pml_Hi = 0 0 1
warpx.mag_time_scheme_order = 1 # default 1
warpx.mag_M_normalization = 1 # 1 is saturated
warpx.mag_LLG_coupling = 1
particles.nspecies = 0

algo.em_solver_medium = macroscopic # vacuum/macroscopic

algo.macroscopic_sigma_method = laxwendroff # laxwendroff or backwardeuler

macroscopic.sigma_init_style = "parse_sigma_function" # parse or "constant"
macroscopic.sigma_function(x,y,z) = "0.0"

macroscopic.epsilon_init_style = "parse_epsilon_function" # parse or "constant"
macroscopic.epsilon_function(x,y,z) = "8.8541878128e-12"

macroscopic.mu_init_style = "parse_mu_function" # parse or "constant"
macroscopic.mu_function(x,y,z) = "1.25663706212e-06"

#unit conversion: 1 Gauss = (1000/4pi) A/m
macroscopic.mag_Ms_init_style = "parse_mag_Ms_function" # parse or "constant"
macroscopic.mag_Ms_function(x,y,z) = "1.4e5 * (z<=h) + 0.1 * (z>h)" # in unit A/m, equal to 1750 Gauss; Ms must be nonzero for LLG

macroscopic.mag_alpha_init_style = "parse_mag_alpha_function" # parse or "constant"
macroscopic.mag_alpha_function(x,y,z) = "0.0058 * (z<=h) + 0 * (z>h)" # alpha is unitless, calculated from linewidth Delta_H = 40 Oersted

macroscopic.mag_gamma_init_style = "parse_mag_gamma_function" # parse or "constant"
macroscopic.mag_gamma_function(x,y,z) = "-1.759e11 * (z<=h) + 0 * (z>h)" # gyromagnetic ratio is constant for electrons in all materials

macroscopic.mag_max_iter = 100 # maximum number of M iteration in each time step
macroscopic.mag_tol = 1.e-6 # M magnitude relative error tolerance compared to previous iteration
macroscopic.mag_normalized_error = 0.1 # if M magnitude relatively changes more than this value, raise a red flag

#################################
############ FIELDS #############
#################################

warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = 0.
warpx.Ey_external_grid_function(x,y,z) = 0.
warpx.Ez_external_grid_function(x,y,z) = 0.

warpx.E_excitation_on_grid_style = "parse_E_excitation_grid_function"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ey_excitation_grid_function(x,y,z,t) = "(exp((t-3*TP)**2/(2*TP**2))*cos(2*pi*c/wavelength*t)) * (z>=h && z<h+1.0e-4)"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0.0"

warpx.H_ext_grid_init_style = parse_H_ext_grid_function
warpx.Hx_external_grid_function(x,y,z)= 0.
warpx.Hy_external_grid_function(x,y,z) = 0.
warpx.Hz_external_grid_function(x,y,z) = 0.

#unit conversion: 1 Gauss = 1 Oersted = (1000/4pi) A/m
#calculation of H_bias: H_bias (oe) = frequency / 2.8e6

warpx.H_bias_ext_grid_init_style = parse_H_bias_ext_grid_function
warpx.Hx_bias_external_grid_function(x,y,z)= 0.
warpx.Hy_bias_external_grid_function(x,y,z)= "9470.0 * (z<=h) + 0 * (z>h)" # in A/m, equal to 120 Oersted
warpx.Hz_bias_external_grid_function(x,y,z)= 0. 

warpx.M_ext_grid_init_style = parse_M_ext_grid_function
warpx.Mx_external_grid_function(x,y,z)= 0.
warpx.My_external_grid_function(x,y,z)= "1.4e5 * (z<=h) + 0.1 * (z>h)" 
warpx.Mz_external_grid_function(x,y,z) = 0.

#Diagnostics
diagnostics.diags_names = plt
plt.period = 1
plt.diag_type = Full
#plt.fields_to_plot = Ex Ey Ez Bx By Bz
plt.fields_to_plot = Ex Ey Ez Hx Hy Hz Bx By Bz Mx_xface My_xface Mz_xface Mx_yface My_yface Mz_yface Mx_zface My_zface Mz_zface
plt.plot_raw_fields = 0

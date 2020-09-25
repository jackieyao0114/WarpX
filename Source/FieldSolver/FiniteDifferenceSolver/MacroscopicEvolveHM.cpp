/* 
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>

using namespace amrex;

/**
 * \brief Update H and M fields with iterative correction, over one time step
 */

#ifdef WARPX_MAG_LLG
// update M field over one timestep

void FiniteDifferenceSolver::MacroscopicEvolveHM(
    // The MField here is a vector of three multifabs, with M on each face, and each multifab is a three-component multifab.
    // Each M-multifab has three components, one for each component in x, y, z. (All multifabs are four dimensional, (i,j,k,n)), where, n=1 for E, B, but, n=3 for M_xface, M_yface, M_zface
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield,            // H Maxwell
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties)
{

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveHMCartesian<CartesianYeeAlgorithm>(Mfield, Hfield, H_biasfield, Efield, dt, macroscopic_properties);
    }
    else
    {
        amrex::Abort("Only yee algorithm is compatible for H and M updates.");
    }
} // closes function EvolveM
#endif

#ifdef WARPX_MAG_LLG
template <typename T_Algo>
void FiniteDifferenceSolver::MacroscopicEvolveHMCartesian(
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield,
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield,            // H Maxwell
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield, // H bias
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const &macroscopic_properties)
{

    auto &warpx = WarpX::GetInstance();
    int coupling = warpx.mag_LLG_coupling;
    int M_normalization = warpx.mag_M_normalization;

    // build temporary vector<multifab,3> Mfield_prev_iter, Mfield_error, a_temp, a_temp_static, b_temp_static
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > Mfield_prev_iter; // Mfield_prev_iter is M(n+1) of the (r-1)th iteration
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > Mfield_old; // Mfield_old is M(n)
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > Mfield_error; // The error of the M field between the two iterations

    for (int i = 0; i < 3; i++)
    {
        // Mfield_prev_iter is M(n+1) of the (r-1)th iteration
        Mfield_prev_iter[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        // Mfield_old is M(n)
        Mfield_old[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        // Mfield_error is the error of the M field between the two iterations
        Mfield_error[i].reset(new MultiFab(Mfield[i]->boxArray(), Mfield[i]->DistributionMap(), 3, Mfield[i]->nGrow()));
        // initialize Mfield_prev_iter
        MultiFab::Copy(*Mfield_prev_iter[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
        // initialize Mfield_old
        MultiFab::Copy(*Mfield_old[i], *Mfield[i], 0, 0, 3, Mfield[i]->nGrow());
    }

    // obtain the maximum relative amount we let M deviate from Ms before aborting
    amrex::Real mag_normalized_error = macroscopic_properties->getmag_normalized_error();

    // relative tolerance stopping criteria for the iterative algorithm
    // initialize M_max_iter, M_iter, M_tol, M_iter_error
    amrex::Real M_tol = macroscopic_properties->getmag_tol();
    // maximum number of iterations allowed
    int M_max_iter = macroscopic_properties->getmag_max_iter();
    // actual number of iteration conducted
    int M_iter = 0;
    int stop_iter = 0;

    // calculate the maximum absolute value of the Mfield_prev_iter
    amrex::GpuArray<amrex::Real, 3> Mfield_prev_iter_max;
    for (int i = 0; i < 3; i++)
    {
        Mfield_prev_iter_max[i] = std::max(amrex::Math::abs((*Mfield_prev_iter[i]).max(i, 0)), amrex::Math::abs((*Mfield_prev_iter[i]).min(i, 0)));
    }

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

    // begin the iteration
    while (!stop_iter)
    {
        for (MFIter mfi(*Hfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {

            // Extract field data for this grid/tile
            Array4<Real> const &Hx = Hfield[0]->array(mfi);
            Array4<Real> const &Hy = Hfield[1]->array(mfi);
            Array4<Real> const &Hz = Hfield[2]->array(mfi);
            Array4<Real> const &Ex = Efield[0]->array(mfi);
            Array4<Real> const &Ey = Efield[1]->array(mfi);
            Array4<Real> const &Ez = Efield[2]->array(mfi);
            Array4<Real> const &M_xface = Mfield[0]->array(mfi);           // note M_xface include x,y,z components at |_x faces
            Array4<Real> const &M_yface = Mfield[1]->array(mfi);           // note M_yface include x,y,z components at |_y faces
            Array4<Real> const &M_zface = Mfield[2]->array(mfi);           // note M_zface include x,y,z components at |_z faces
            Array4<Real> const &M_xface_old = Mfield_old[0]->array(mfi); // note M_xface_old include x,y,z components at |_x faces
            Array4<Real> const &M_yface_old = Mfield_old[1]->array(mfi); // note M_yface_old include x,y,z components at |_y faces
            Array4<Real> const &M_zface_old = Mfield_old[2]->array(mfi); // note M_zface_old include x,y,z components at |_z faces

            // Extract stencil coefficients
            Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // Extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect());
            Box const &tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

            // Loop over the cells and update the fields
            amrex::ParallelFor(
                tbx, tby, tbz,

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Hx(i, j, k) += 1 / PhysConst::mu0 * dt * T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k) 
                                 - 1 / PhysConst::mu0 * dt * T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k) 
                                 - M_xface(i, j, k, 0) + M_xface_old(i, j, k, 0);
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Hy(i, j, k) += 1 / PhysConst::mu0 * dt * T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k) 
                                 - 1 / PhysConst::mu0 * dt * T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k) 
                                 - M_yface(i, j, k, 1) + M_yface_old(i, j, k, 1);
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Hz(i, j, k) += 1 / PhysConst::mu0 * dt * T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k) 
                                 - 1 / PhysConst::mu0 * dt * T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k) 
                                 - M_zface(i, j, k, 2) + M_zface_old(i, j, k, 2);
                }

            );
        }

        for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
        {
            auto &mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
            auto &mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
            auto &mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
            // extract material properties
            Array4<Real> const &mag_Ms_arr = mag_Ms_mf.array(mfi);
            Array4<Real> const &mag_alpha_arr = mag_alpha_mf.array(mfi);
            Array4<Real> const &mag_gamma_arr = mag_gamma_mf.array(mfi);

            // extract field data
            Array4<Real> const &Hx = Hfield[0]->array(mfi);
            Array4<Real> const &Hy = Hfield[1]->array(mfi);
            Array4<Real> const &Hz = Hfield[2]->array(mfi);
            Array4<Real> const &M_xface = Mfield[0]->array(mfi);           // note M_xface include x,y,z components at |_x faces
            Array4<Real> const &M_yface = Mfield[1]->array(mfi);           // note M_yface include x,y,z components at |_y faces
            Array4<Real> const &M_zface = Mfield[2]->array(mfi);           // note M_zface include x,y,z components at |_z faces
            Array4<Real> const &M_xface_prev_iter = Mfield_prev_iter[0]->array(mfi); // note M_xface_prev_iter include x,y,z components at |_x faces
            Array4<Real> const &M_yface_prev_iter = Mfield_prev_iter[1]->array(mfi); // note M_yface_prev_iter include x,y,z components at |_y faces
            Array4<Real> const &M_zface_prev_iter = Mfield_prev_iter[2]->array(mfi); // note M_zface_prev_iter include x,y,z components at |_z faces
            Array4<Real> const &M_xface_old = Mfield_old[0]->array(mfi); // note M_xface_old include x,y,z components at |_x faces
            Array4<Real> const &M_yface_old = Mfield_old[1]->array(mfi); // note M_yface_old include x,y,z components at |_y faces
            Array4<Real> const &M_zface_old = Mfield_old[2]->array(mfi); // note M_zface_old include x,y,z components at |_z faces
            Array4<Real> const &M_error_xface = Mfield_error[0]->array(mfi);
            Array4<Real> const &M_error_yface = Mfield_error[1]->array(mfi);
            Array4<Real> const &M_error_zface = Mfield_error[2]->array(mfi);
            Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi); // Hx_bias is the x component at |_x faces
            Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi); // Hy_bias is the y component at |_y faces
            Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi); // Hz_bias is the z component at |_z faces

            // extract stencil coefficients
            Real const *const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            Real const *const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            Real const *const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect()); /* just define which grid type */
            Box const &tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(
                tbx, tby, tbz,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx_bias);
                    Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy_bias);
                    Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz_bias);
                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(1, 0, 0), Hx);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(1, 0, 0), Hy);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(1, 0, 0), Hz);
                    }

                    // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
                    // keep the interpolation. The IntVect is (1,0,0) to interpolate values to the x-face.
                    Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1, 0, 0), mag_gamma_arr);

                    Real M_magnitude = (M_normalization == 0) ? 
                                       std::sqrt(std::pow(M_xface(i, j, k, 0), 2.0) + std::pow(M_xface(i, j, k, 1), 2.0) + std::pow(M_xface(i, j, k, 2), 2.0)) 
                                     : MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1, 0, 0), mag_Ms_arr);
                    Real Gil_damp = PhysConst::mu0 * mag_gamma_interp * MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1, 0, 0), mag_alpha_arr) / M_magnitude;

                    // now you have access to use M_xface(i,j,k,0) M_xface(i,j,k,1), M_xface(i,j,k,2), Hx(i,j,k), Hy, Hz on the RHS of these update lines below
                    // x component on x-faces of grid
                    M_xface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff) 
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 1) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff) - M_xface_old(i, j, k, 2) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff));

                    // y component on x-faces of grid
                    M_xface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff) 
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 2) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff) - M_xface_old(i, j, k, 0) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff));

                    // z component on x-faces of grid
                    M_xface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_xface_old(i, j, k, 0) * Hy_eff - M_xface_old(i, j, k, 1) * Hx_eff) 
                                         + dt * Gil_damp * (M_xface_old(i, j, k, 0) * (M_xface_old(i, j, k, 2) * Hx_eff - M_xface_old(i, j, k, 0) * Hz_eff) - M_xface_old(i, j, k, 1) * (M_xface_old(i, j, k, 1) * Hz_eff - M_xface_old(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_xface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2.0) + std::pow(M_xface(i, j, k, 1), 2.0) + std::pow(M_xface(i, j, k, 2), 2.0))
                                                         / MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1, 0, 0), mag_Ms_arr);

                    if (M_normalization > 0)
                    {
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                            amrex::Abort("Exceed the normalized error of the M_xface field");
                        }
                        // normalize the M_xface field
                        M_xface(i, j, k, 0) /= M_magnitude_normalized;
                        M_xface(i, j, k, 1) /= M_magnitude_normalized;
                        M_xface(i, j, k, 2) /= M_magnitude_normalized;
                    }
                    else if (M_normalization == 0)
                    {
                        // check the normalized error
                        if (M_magnitude_normalized > 1._rt + mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized,
                                   MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(1, 0, 0), mag_Ms_arr));
                            amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                        }
                        else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error)
                        {
                            // normalize the M_xface field
                            M_xface(i, j, k, 0) /= M_magnitude_normalized;
                            M_xface(i, j, k, 1) /= M_magnitude_normalized;
                            M_xface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                    }
                    // calculate M_error_xface
                    // x component on x-faces of grid
                    M_error_xface(i, j, k, 0) = std::abs((M_xface(i, j, k, 0) - M_xface_prev_iter(i, j, k, 0))) / Mfield_prev_iter_max[0];

                    // y component on x-faces of grid
                    M_error_xface(i, j, k, 1) = std::abs((M_xface(i, j, k, 1) - M_xface_prev_iter(i, j, k, 1))) / Mfield_prev_iter_max[1];

                    // z component on x-faces of grid
                    M_error_xface(i, j, k, 2) = std::abs((M_xface(i, j, k, 2) - M_xface_prev_iter(i, j, k, 2))) / Mfield_prev_iter_max[2];
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx_bias);
                    Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy_bias);
                    Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz_bias);
                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 1, 0), Hx);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 1, 0), Hy);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 1, 0), Hz);
                    }

                    // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
                    // keep the interpolation. The IntVect is (0,1,0) to interpolate values to the y-face.
                    Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 1, 0), mag_gamma_arr);

                    Real M_magnitude = (M_normalization == 0) ? 
                                    std::sqrt(std::pow(M_yface(i, j, k, 0), 2.0) + std::pow(M_yface(i, j, k, 1), 2.0) + std::pow(M_yface(i, j, k, 2), 2.0)) 
                                  : MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 1, 0), mag_Ms_arr);
                    Real Gil_damp = PhysConst::mu0 * mag_gamma_interp * MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 1, 0), mag_alpha_arr) / M_magnitude;

                    // x component on y-faces of grid
                    M_yface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff) 
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 1) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff) - M_yface_old(i, j, k, 2) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff));

                    // y component on y-faces of grid
                    M_yface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff) 
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 2) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff) - M_yface_old(i, j, k, 0) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff));

                    // z component on y-faces of grid
                    M_yface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_yface_old(i, j, k, 0) * Hy_eff - M_yface_old(i, j, k, 1) * Hx_eff) 
                                         + dt * Gil_damp * (M_yface_old(i, j, k, 0) * (M_yface_old(i, j, k, 2) * Hx_eff - M_yface_old(i, j, k, 0) * Hz_eff) - M_yface_old(i, j, k, 1) * (M_yface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_yface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2.0) + std::pow(M_yface(i, j, k, 1), 2.0) + std::pow(M_yface(i, j, k, 2), 2.0))
                                                         / MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 1, 0), mag_Ms_arr);

                    if (M_normalization > 0)
                    {
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                            amrex::Abort("Exceed the normalized error of the M_yface field");
                        }
                        // normalize the M_yface field
                        M_yface(i, j, k, 0) /= M_magnitude_normalized;
                        M_yface(i, j, k, 1) /= M_magnitude_normalized;
                        M_yface(i, j, k, 2) /= M_magnitude_normalized;
                    }
                    else if (M_normalization == 0)
                    {
                        // check the normalized error
                        if (M_magnitude_normalized > 1._rt + mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized,
                                   MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 1, 0), mag_Ms_arr));
                            amrex::Abort("Caution: Unsaturated material has M_yface exceeding the saturation magnetization");
                        }
                        else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error)
                        {
                            // normalize the M_yface field
                            M_yface(i, j, k, 0) /= M_magnitude_normalized;
                            M_yface(i, j, k, 1) /= M_magnitude_normalized;
                            M_yface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                    }
                    // calculate M_error_yface
                    // x component on y-faces of grid
                    M_error_yface(i, j, k, 0) = std::abs((M_yface(i, j, k, 0) - M_yface_prev_iter(i, j, k, 0))) / Mfield_prev_iter_max[0];

                    // y component on y-faces of grid
                    M_error_yface(i, j, k, 1) = std::abs((M_yface(i, j, k, 1) - M_yface_prev_iter(i, j, k, 1))) / Mfield_prev_iter_max[1];

                    // z component on y-faces of grid
                    M_error_yface(i, j, k, 2) = std::abs((M_yface(i, j, k, 2) - M_yface_prev_iter(i, j, k, 2))) / Mfield_prev_iter_max[2];
                },

                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    Real Hx_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx_bias);
                    Real Hy_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy_bias);
                    Real Hz_eff = MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz_bias);

                    if (coupling == 1)
                    {
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)

                        // H_maxwell
                        Hx_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(1, 0, 0), amrex::IntVect(0, 0, 1), Hx);
                        Hy_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 1, 0), amrex::IntVect(0, 0, 1), Hy);
                        Hz_eff += MacroscopicProperties::face_avg_to_face(i, j, k, 0, amrex::IntVect(0, 0, 1), amrex::IntVect(0, 0, 1), Hz);
                    }

                    // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
                    // keep the interpolation. The IntVect is (0,0,1) to interpolate values to the z-face.
                    Real mag_gamma_interp = MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 0, 1), mag_gamma_arr);

                    Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface(i, j, k, 0), 2.0_rt) + std::pow(M_zface(i, j, k, 1), 2.0_rt) + std::pow(M_zface(i, j, k, 2), 2.0_rt)) : MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 0, 1), mag_Ms_arr);
                    Real Gil_damp = PhysConst::mu0 * mag_gamma_interp * MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 0, 1), mag_alpha_arr) / M_magnitude;

                    // x component on z-faces of grid
                    M_zface(i, j, k, 0) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff) 
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 1) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff) - M_zface_old(i, j, k, 2) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff));

                    // y component on z-faces of grid
                    M_zface(i, j, k, 1) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff) 
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 2) * (M_zface_old(i, j, k, 1) * Hz_eff - M_zface_old(i, j, k, 2) * Hy_eff) - M_zface_old(i, j, k, 0) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff));

                    // z component on z-faces of grid
                    M_zface(i, j, k, 2) += dt * (PhysConst::mu0 * mag_gamma_interp) * (M_zface_old(i, j, k, 0) * Hy_eff - M_zface_old(i, j, k, 1) * Hx_eff) 
                                         + dt * Gil_damp * (M_zface_old(i, j, k, 0) * (M_zface_old(i, j, k, 2) * Hx_eff - M_zface_old(i, j, k, 0) * Hz_eff) - M_zface_old(i, j, k, 1) * (M_zface_old(i, j, k, 1) * Hz_eff - M_yface_old(i, j, k, 2) * Hy_eff));

                    // temporary normalized magnitude of M_zface field at the fixed point
                    // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                    amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2.0_rt) + std::pow(M_zface(i, j, k, 1), 2.0_rt) + std::pow(M_zface(i, j, k, 2), 2.0_rt))
                                                         / MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 0, 1), mag_Ms_arr);

                    if (M_normalization > 0)
                    {
                        // check the normalized error
                        if (amrex::Math::abs(1._rt - M_magnitude_normalized) > mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, mag_normalized_error=%f\n", M_magnitude_normalized, mag_normalized_error);
                            amrex::Abort("Exceed the normalized error of the M_zface field");
                        }
                        // normalize the M_zface field
                        M_zface(i, j, k, 0) /= M_magnitude_normalized;
                        M_zface(i, j, k, 1) /= M_magnitude_normalized;
                        M_zface(i, j, k, 2) /= M_magnitude_normalized;
                    }
                    else if (M_normalization == 0)
                    {
                        // check the normalized error
                        if (M_magnitude_normalized > 1._rt + mag_normalized_error)
                        {
                            printf("i = %d, j=%d, k=%d\n", i, j, k);
                            printf("M_magnitude_normalized = %f, Ms = %f\n", M_magnitude_normalized,
                                   MacroscopicProperties::macro_avg_to_face(i, j, k, amrex::IntVect(0, 0, 1), mag_Ms_arr));
                            amrex::Abort("Caution: Unsaturated material has M_zface exceeding the saturation magnetization");
                        }
                        else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= 1._rt + mag_normalized_error)
                        {
                            // normalize the M_zface field
                            M_zface(i, j, k, 0) /= M_magnitude_normalized;
                            M_zface(i, j, k, 1) /= M_magnitude_normalized;
                            M_zface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                    }
                    // calculate M_error_zface
                    // x component on z-faces of grid
                    M_error_zface(i, j, k, 0) = amrex::Math::abs((M_zface(i, j, k, 0) - M_zface_prev_iter(i, j, k, 0))) / Mfield_prev_iter_max[0];

                    // y component on z-faces of grid
                    M_error_zface(i, j, k, 1) = amrex::Math::abs((M_zface(i, j, k, 1) - M_zface_prev_iter(i, j, k, 1))) / Mfield_prev_iter_max[1];

                    // z component on z-faces of grid
                    M_error_zface(i, j, k, 2) = amrex::Math::abs((M_zface(i, j, k, 2) - M_zface_prev_iter(i, j, k, 2))) / Mfield_prev_iter_max[2];
                });
        }

        // Check the error between Mfield and Mfield_prev_iter 
        amrex::Real M_iter_maxerror = -1.0;
        for (int i = 0; i < 1; i++){
            for (int j = 0; j < 3; j++){
                Real M_iter_error = Mfield_error[i]->norm0(j);
                if(M_iter_error >= M_iter_maxerror) {
                     M_iter_maxerror = M_iter_error;
                }
            }
        }

        // Decide whether another iteration is needed
        if (M_iter_maxerror <= M_tol) {

            stop_iter = 1;

        } else {
            // Copy Mfield to Mfield_prev_iterious and re-calculate Mfield_prev_iter_max
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(*Mfield_prev_iter[i],*Mfield[i],0,0,3,Mfield[i]->nGrow());
                Mfield_prev_iter_max[i] = std::max(amrex::Math::abs((*Mfield_prev_iter[i]).max(i,0)),amrex::Math::abs((*Mfield_prev_iter[i]).min(i,0)));
            }
        }

        if (M_iter >= M_max_iter) {
            amrex::Abort("The M_iter exceeds the M_max_iter");
            amrex::Print() << "The M_iter = " << M_iter << " exceeds the M_max_iter = " << M_max_iter << std::endl;
        } else {
            M_iter++;
            amrex::Print() << "Finish " << M_iter << " times iteration with M_iter_maxerror = " << M_iter_maxerror << " and M_tol = " << M_tol << std::endl;
        }
    } // end of iteration
}
#endif
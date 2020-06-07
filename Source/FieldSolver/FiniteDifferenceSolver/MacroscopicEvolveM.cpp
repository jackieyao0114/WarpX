/* copyright
blank
*/

#include "Utils/WarpXAlgorithmSelection.H"
#include "FiniteDifferenceSolver.H"
#include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"

#include "Utils/WarpXConst.H"
#include <AMReX_Gpu.H>

using namespace amrex;

// update M field over one timestep

void FiniteDifferenceSolver::MacroscopicEvolveM (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Mfield, // Mfield contains three components MultiFab
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Hfield, // added another argument Hfield to include Heff
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias 
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties) {

    /* if (m_do_nodal) {

        EvolveMCartesian <CartesianNodalAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        EvolveMCartesian <CartesianYeeAlgorithm> ( Mfield, Bfield, dt );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        EvolveMCartesian <CartesianCKCAlgorithm> ( Mfield, Bfield, dt );
    }
    else
    {
        amrex::Abort("Unknown algorithm");
    } */

    if (m_fdtd_algo == MaxwellSolverAlgo::Yee)
    {
        MacroscopicEvolveMCartesian <CartesianYeeAlgorithm> (Mfield, Hfield, H_biasfield, Bfield, dt, macroscopic_properties);
    }
    else {
       amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
    } // closes function EvolveM

    template<typename T_Algo>
    void FiniteDifferenceSolver::MacroscopicEvolveMCartesian (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Hfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& H_biasfield, // H bias
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        amrex::Real const dt,
        std::unique_ptr<MacroscopicProperties> const& macroscopic_properties )
    {
        // static constexpr amrex::Real alpha = 1e-4;
        // static constexpr amrex::Real Ms = 1e4;
        // Real constexpr cons1 = -mag_gamma_interp; // should be mu0*gamma, mu0 is absorbed by B used in this case
        // Real constexpr cons2 = -cons1*alpha/Ms; // factor of the second term in scalar LLG

	    for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
        {
          auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
          auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
          auto& mag_gamma_mf = macroscopic_properties->getmag_gamma_mf();
          // exctract material properties
          Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
          Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
          Array4<Real> const& mag_gamma_arr = mag_gamma_mf.array(mfi);

            // extract field data
            Array4<Real> const& Mx = Mfield[0]->array(mfi); // note Mx are x,y,z components at |_x faces
            Array4<Real> const& My = Mfield[1]->array(mfi);
            Array4<Real> const& Mz = Mfield[2]->array(mfi);
            Array4<Real> const& Hx = Hfield[0]->array(mfi);
            Array4<Real> const& Hy = Hfield[1]->array(mfi);
            Array4<Real> const& Hz = Hfield[2]->array(mfi);
            Array4<Real> const& Hx_bias = H_biasfield[0]->array(mfi);
            Array4<Real> const& Hy_bias = H_biasfield[1]->array(mfi);
            Array4<Real> const& Hz_bias = H_biasfield[2]->array(mfi);
            Array4<Real> const& Bx = Bfield[0]->array(mfi);
            Array4<Real> const& By = Bfield[1]->array(mfi);
            Array4<Real> const& Bz = Bfield[2]->array(mfi);

            // extract stencil coefficients
            Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            int const n_coefs_x = m_stencil_coefs_x.size();
            Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            int const n_coefs_y = m_stencil_coefs_y.size();
            Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = m_stencil_coefs_z.size();

            // extract tileboxes for which to loop
            Box const& tbx = mfi.tilebox(Hfield[0]->ixType().toIntVect()); /* just define which grid type */
            Box const& tby = mfi.tilebox(Hfield[1]->ixType().toIntVect());
            Box const& tbz = mfi.tilebox(Hfield[2]->ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // figure out if interpolation of Mx, My and Mz is needed ???
              Hx(i, j, k) = Bx(i, j, k)/PhysConst::mu0 - Mx(i, j, k, 0);
              Hy(i, j, k) = By(i, j, k)/PhysConst::mu0 - Mx(i, j, k, 1);
              Hz(i, j, k) = Bz(i, j, k)/PhysConst::mu0 - Mx(i, j, k, 2);

              // when working on Mx(i,j,k, 0:2) we have direct access to Mx(i,j,k,0:2) and Hx(i,j,k)
              // Hy and Hz can be acquired by interpolation
              // H_maxwell
              Real Hy_xtemp = 0.25*(Hy(i,j,k)+Hy(i,j+1,k)+Hy(i-1,j,k)+Hy(i-1,j+1,k));
              Real Hz_xtemp = 0.25*(Hz(i,j,k)+Hz(i-1,j,k)+Hz(i,j,k+1)+Hz(i-1,j,k+1));
              // H_bias
              Real Hy_bias_xtemp = 0.25*(Hy_bias(i,j,k)+Hy_bias(i,j+1,k)+Hy_bias(i-1,j,k)+Hy_bias(i-1,j+1,k));
              Real Hz_bias_xtemp = 0.25*(Hz_bias(i,j,k)+Hz_bias(i-1,j,k)+Hz_bias(i,j,k+1)+Hz_bias(i-1,j,k+1));
              // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy ... (only the first two terms are considered here)
              Real Hx_eff = Hx(i,j,k) + Hx_bias(i,j,k);
              Real Hy_eff = Hy_xtemp + Hy_bias_xtemp;
              Real Hz_eff = Hz_xtemp + Hz_bias_xtemp;

              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              Real Gil_damp = PhysConst::mu0 * 0.25 * (
                                 mag_gamma_arr(i,j,k) * mag_alpha_arr(i,j,k)/mag_Ms_arr(i,j,k)
                                 + mag_gamma_arr(i,j+1,k) * mag_alpha_arr(i,j+1,k)/mag_Ms_arr(i,j+1,k)
                                 + mag_gamma_arr(i,j,k+1) * mag_alpha_arr(i,j,k+1)/mag_Ms_arr(i,j,k+1)
                                 + mag_gamma_arr(i,j+1,k+1) * mag_alpha_arr(i,j+1,k+1)/mag_Ms_arr(i,j+1,k+1));
              
              Real mag_gamma_interp = 0.25 * (mag_gamma_arr(i,j,k) + mag_gamma_arr(i,j+1,k) + mag_gamma_arr(i,j,k+1) + mag_gamma_arr(i,j+1,k+1));
              
              // now you have access to use Mx(i,j,k,0) Mx(i,j,k,1), Mx(i,j,k,2), Hx(i,j,k), Hy, Hz on the RHS of these update lines below

              // x component on x-faces of grid
              Mx(i, j, k, 0) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( Mx(i, j, k, 1) * Hz_eff - Mx(i, j, k, 2) * Hy_eff)
                + dt * Gil_damp * ( Mx(i, j, k, 1) * (Mx(i, j, k, 0) * Hy_eff - Mx(i, j, k, 1) * Hx_eff)
                - Mx(i, j, k, 2) * ( Mx(i, j, k, 2) * Hx_eff - Mx(i, j, k, 0) * Hz_eff));

              // y component on x-faces of grid
              Mx(i, j, k, 1) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( Mx(i, j, k, 2) * Hx_eff - Mx(i, j, k, 0) * Hz_eff)
                + dt * Gil_damp * ( Mx(i, j, k, 2) * (Mx(i, j, k, 1) * Hz_eff - Mx(i, j, k, 2) * Hy_eff)
                - Mx(i, j, k, 0) * ( Mx(i, j, k, 0) * Hy_eff - Mx(i, j, k, 1) * Hx_eff));

              // z component on x-faces of grid
              Mx(i, j, k, 2) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( Mx(i, j, k, 0) * Hy_eff - Mx(i, j, k, 1) * Hx_eff)
                + dt * Gil_damp * ( Mx(i, j, k, 0) * ( Mx(i, j, k, 2) * Hx_eff - Mx(i, j, k, 0) * Hz_eff)
                - Mx(i, j, k, 1) * ( Mx(i, j, k, 1) * Hz_eff - Mx(i, j, k, 2) * Hy_eff));
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on My(i,j,k,0:2) we have direct access to My(i,j,k,0:2) and Hy(i,j,k)
              Real Hx_ytemp = 0.25*(Hx(i,j,k)+Hx(i+1,j,k)+Hx(i,j-1,k)+Hx(i+1,j-1,k));
              Real Hz_ytemp = 0.25*(Hz(i,j,k)+Hz(i,j,k+1)+Hz(i,j-1,k)+Hz(i,j-1,k+1));
              Real Hx_bias_ytemp = 0.25*(Hx_bias(i,j,k)+Hx_bias(i+1,j,k)+Hx_bias(i,j-1,k)+Hx_bias(i+1,j-1,k));
              Real Hz_bias_ytemp = 0.25*(Hz_bias(i,j,k)+Hz_bias(i,j,k+1)+Hz_bias(i,j-1,k)+Hz_bias(i,j-1,k+1));
              Real Hx_eff = Hx_ytemp + Hx_bias_ytemp;
              Real Hy_eff = Hy(i,j,k) + Hy_bias(i,j,k);
              Real Hz_eff = Hz_ytemp + Hz_bias_ytemp;

              Real Gil_damp = PhysConst::mu0 * 0.25 * (mag_gamma_arr(i,j,k) * mag_alpha_arr(i,j,k)/mag_Ms_arr(i,j,k)
                                 + mag_gamma_arr(i+1,j,k) * mag_alpha_arr(i+1,j,k)/mag_Ms_arr(i+1,j,k)
                                 + mag_gamma_arr(i,j,k+1) * mag_alpha_arr(i,j,k+1)/mag_Ms_arr(i,j,k+1)
                                 + mag_gamma_arr(i+1,j,k+1) * mag_alpha_arr(i+1,j,k+1)/mag_Ms_arr(i+1,j,k+1));
              Real mag_gamma_interp = 0.25 * (mag_gamma_arr(i,j,k) + mag_gamma_arr(i+1,j,k) + mag_gamma_arr(i,j,k+1) + mag_gamma_arr(i+1,j,k+1));
              
              // x component on y-faces of grid
              My(i, j, k, 0) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( My(i, j, k, 1) * Hz_eff - My(i, j, k, 2) * Hy_eff)
                + dt * Gil_damp * ( My(i, j, k, 1) * (My(i, j, k, 0) * Hy_eff - My(i, j, k, 1) * Hx_eff)
                - My(i, j, k, 2) * ( My(i, j, k, 2) * Hz_eff - My(i, j, k, 0) * Hz_eff));

              // y component on y-faces of grid
              My(i, j, k, 1) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( My(i, j, k, 2) * Hz_eff - My(i, j, k, 0) * Hz_eff)
                + dt * Gil_damp * ( My(i, j, k, 2) * (My(i, j, k, 1) * Hz_eff - My(i, j, k, 2) * Hy_eff)
                - My(i, j, k, 0) * ( My(i, j, k, 0) * Hy_eff - My(i, j, k, 1) * Hz_eff));

              // z component on y-faces of grid
              My(i, j, k, 2) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( My(i, j, k, 0) * Hy_eff - My(i, j, k, 1) * Hz_eff)
                + dt * Gil_damp * ( My(i, j, k, 0) * ( My(i, j, k, 2) * Hz_eff - My(i, j, k, 0) * Hz_eff)
                - My(i, j, k, 1) * ( My(i, j, k, 1) * Hz_eff - My(i, j, k, 2) * Hy_eff));
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on Mz(i,j,k,0:2) we have direct access to Mz(i,j,k,0:2) and Hz(i,j,k)
              Real Hx_ztemp = 0.25*(Hx(i,j,k)+Hx(i+1,j,k)+Hx(i+1,j,k-1)+Hx(i,j,k-1));
              Real Hy_ztemp = 0.25*(Hy(i,j,k)+Hy(i,j,k-1)+Hy(i,j+1,k)+Hy(i,j+1,k-1));
              Real Hx_bias_ztemp = 0.25*(Hx_bias(i,j,k)+Hx_bias(i+1,j,k)+Hx_bias(i+1,j,k-1)+Hx_bias(i,j,k-1));
              Real Hy_bias_ztemp = 0.25*(Hy_bias(i,j,k)+Hy_bias(i,j,k-1)+Hy_bias(i,j+1,k)+Hy_bias(i,j+1,k-1));
              Real Hx_eff = Hx_ztemp + Hx_bias_ztemp;
              Real Hy_eff = Hy_ztemp + Hy_bias_ztemp;
              Real Hz_eff = Hz(i,j,k) + Hz_bias(i,j,k);

              Real Gil_damp = PhysConst::mu0 * 0.25 * (mag_gamma_arr(i,j,k) * mag_alpha_arr(i,j,k)/mag_Ms_arr(i,j,k)
                                 + mag_gamma_arr(i+1,j,k) * mag_alpha_arr(i+1,j,k)/mag_Ms_arr(i+1,j,k)
                                 + mag_gamma_arr(i,j+1,k) * mag_alpha_arr(i,j+1,k)/mag_Ms_arr(i,j+1,k)
                                 + mag_gamma_arr(i+1,j+1,k) * mag_alpha_arr(i+1,j+1,k)/mag_Ms_arr(i+1,j+1,k));
              Real mag_gamma_interp = 0.25 * (mag_gamma_arr(i,j,k) + mag_gamma_arr(i+1,j,k) + mag_gamma_arr(i,j+1,k) + mag_gamma_arr(i+1,j+1,k));
              
              // x component on z-faces of grid
              Mz(i, j, k, 0) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( Mz(i, j, k, 1) * Hz_eff - Mz(i, j, k, 2) * Hy_eff)
                + dt * Gil_damp * ( Mz(i, j, k, 1) * (Mz(i, j, k, 0) * Hy_eff - Mz(i, j, k, 1) * Hx_eff)
                - Mz(i, j, k, 2) * ( Mz(i, j, k, 2) * Hx_eff - Mz(i, j, k, 0) * Hz_eff));

              // y component on z-faces of grid
              Mz(i, j, k, 1) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( Mz(i, j, k, 2) * Hx_eff - Mz(i, j, k, 0) * Hz_eff)
                + dt * Gil_damp * ( Mz(i, j, k, 2) * (Mz(i, j, k, 1) * Hz_eff - Mz(i, j, k, 2) * Hy_eff)
                - Mz(i, j, k, 0) * ( Mz(i, j, k, 0) * Hy_eff - Mz(i, j, k, 1) * Hx_eff));

              // z component on z-faces of grid
              Mz(i, j, k, 2) += dt * (-PhysConst::mu0 * mag_gamma_interp) * ( Mz(i, j, k, 0) * Hy_eff - Mz(i, j, k, 1) * Hx_eff)
                + dt * Gil_damp * ( Mz(i, j, k, 0) * ( Mz(i, j, k, 2) * Hx_eff - Mz(i, j, k, 0) * Hz_eff)
                - Mz(i, j, k, 1) * ( Mz(i, j, k, 1) * Hz_eff - My(i, j, k, 2) * Hy_eff));
            }
            );
        }
    }

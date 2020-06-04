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
        MacroscopicEvolveMCartesian <CartesianYeeAlgorithm> (Mfield, Hfield, Bfield, dt, macroscopic_properties);
    }
    else {
       amrex::Abort("Only yee algorithm is compatible for M updates.");
    }
    } // closes function EvolveM

    template<typename T_Algo>
    void FiniteDifferenceSolver::MacroscopicEvolveMCartesian (
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Mfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > & Hfield,
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
        amrex::Real const dt,
        std::unique_ptr<MacroscopicProperties> const& macroscopic_properties )
    {

	    for (MFIter mfi(*Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) /* remember to FIX */
        {
          auto& mag_Ms_mf = macroscopic_properties->getmag_Ms_mf();
          auto& mag_alpha_mf = macroscopic_properties->getmag_alpha_mf();
          // exctract material properties
          Array4<Real> const& mag_Ms_arr = mag_Ms_mf.array(mfi);
          Array4<Real> const& mag_alpha_arr = mag_alpha_mf.array(mfi);
          
            // extract field data
            Array4<Real> const& Mx = Mfield[0]->array(mfi); // note Mx are x,y,z components at |_x faces
            Array4<Real> const& My = Mfield[1]->array(mfi);
            Array4<Real> const& Mz = Mfield[2]->array(mfi);
            Array4<Real> const& Hx = Hfield[0]->array(mfi);
            Array4<Real> const& Hy = Hfield[1]->array(mfi);
            Array4<Real> const& Hz = Hfield[2]->array(mfi);
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
              Real Hy_xtemp = 0.25*(Hy(i,j,k)+Hy(i,j+1,k)+Hy(i-1,j,k)+Hy(i-1,j+1,k));
              Real Hz_xtemp = 0.25*(Hz(i,j,k)+Hz(i-1,j,k)+Hz(i,j,k+1)+Hz(i-1,j,k+1));
              // magnetic material properties mag_alpha and mag_Ms are defined at cell nodes
              Real Gil_damp = PhysConst::mu0 * PhysConst::mag_gamma * 0.25 * (mag_alpha_arr(i,j,k)/mag_Ms_arr(i,j,k)
                                 +mag_alpha_arr(i,j+1,k)/mag_Ms_arr(i,j+1,k)
                                 +mag_alpha_arr(i,j,k+1)/mag_Ms_arr(i,j,k+1)
                                 +mag_alpha_arr(i,j+1,k+1)/mag_Ms_arr(i,j+1,k+1));
              // now you have access to use Mx(i,j,k,0) Mx(i,j,k,1), Mx(i,j,k,2), Hx(i,j,k), Hy, Hz on the RHS of these update lines below

              // x component on x-faces of grid
              Mx(i, j, k, 0) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( Mx(i, j, k, 1) * Hz_xtemp - Mx(i, j, k, 2) * Hy_xtemp)
                + dt * Gil_damp * ( Mx(i, j, k, 1) * (Mx(i, j, k, 0) * Hy_xtemp - Mx(i, j, k, 1) * Hx(i, j, k))
                - Mx(i, j, k, 2) * ( Mx(i, j, k, 2) * Hx(i, j, k) - Mx(i, j, k, 0) * Hz_xtemp));

              // y component on x-faces of grid
              Mx(i, j, k, 1) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( Mx(i, j, k, 2) * Hx(i, j, k) - Mx(i, j, k, 0) * Hz_xtemp)
                + dt * Gil_damp * ( Mx(i, j, k, 2) * (Mx(i, j, k, 1) * Hz_xtemp - Mx(i, j, k, 2) * Hy_xtemp)
                - Mx(i, j, k, 0) * ( Mx(i, j, k, 0) * Hy_xtemp - Mx(i, j, k, 1) * Hx(i, j, k)));

              // z component on x-faces of grid
              Mx(i, j, k, 2) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( Mx(i, j, k, 0) * Hy_xtemp - Mx(i, j, k, 1) * Hx(i, j, k))
                + dt * Gil_damp * ( Mx(i, j, k, 0) * ( Mx(i, j, k, 2) * Hx(i, j, k) - Mx(i, j, k, 0) * Hz_xtemp)
                - Mx(i, j, k, 1) * ( Mx(i, j, k, 1) * Hz_xtemp - Mx(i, j, k, 2) * Hy_xtemp));
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on My(i,j,k,0:2) we have direct access to My(i,j,k,0:2) and Hy(i,j,k)
              Real Hx_ytemp = 0.25*(Hx(i,j,k)+Hx(i+1,j,k)+Hx(i,j-1,k)+Hx(i+1,j-1,k));
              Real Hz_ytemp = 0.25*(Hz(i,j,k)+Hz(i,j,k+1)+Hz(i,j-1,k)+Hz(i,j-1,k+1));
              Real Gil_damp = PhysConst::mu0 * PhysConst::mag_gamma*0.25*(mag_alpha_arr(i,j,k)/mag_Ms_arr(i,j,k)
                                 +mag_alpha_arr(i+1,j,k)/mag_Ms_arr(i+1,j,k)
                                 +mag_alpha_arr(i,j,k+1)/mag_Ms_arr(i,j,k+1)
                                 +mag_alpha_arr(i+1,j,k+1)/mag_Ms_arr(i+1,j,k+1));
              // x component on y-faces of grid
              My(i, j, k, 0) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( My(i, j, k, 1) * Hz_ytemp - My(i, j, k, 2) * Hy(i, j, k))
                + dt * Gil_damp * ( My(i, j, k, 1) * (My(i, j, k, 0) * Hy(i, j, k) - My(i, j, k, 1) * Hx(i, j, k))
                - My(i, j, k, 2) * ( My(i, j, k, 2) * Hx_ytemp - My(i, j, k, 0) * Hz_ytemp));

              // y component on y-faces of grid
              My(i, j, k, 1) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( My(i, j, k, 2) * Hx_ytemp - My(i, j, k, 0) * Hz_ytemp)
                + dt * Gil_damp * ( My(i, j, k, 2) * (My(i, j, k, 1) * Hz_ytemp - My(i, j, k, 2) * Hy(i, j, k))
                - My(i, j, k, 0) * ( My(i, j, k, 0) * Hy(i, j, k) - My(i, j, k, 1) * Hx_ytemp));

              // z component on y-faces of grid
              My(i, j, k, 2) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( My(i, j, k, 0) * Hy(i, j, k) - My(i, j, k, 1) * Hx_ytemp)
                + dt * Gil_damp * ( My(i, j, k, 0) * ( My(i, j, k, 2) * Hx_ytemp - My(i, j, k, 0) * Hz_ytemp)
                - My(i, j, k, 1) * ( My(i, j, k, 1) * Hz_ytemp - My(i, j, k, 2) * Hy(i, j, k)));
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

              // when working on Mz(i,j,k,0:2) we have direct access to Mz(i,j,k,0:2) and Hz(i,j,k)
              Real Hx_ztemp = 0.25*(Hx(i,j,k)+Hx(i+1,j,k)+Hx(i+1,j,k-1)+Hx(i,j,k-1));
              Real Hy_ztemp = 0.25*(Hy(i,j,k)+Hy(i,j,k-1)+Hy(i,j+1,k)+Hy(i,j+1,k-1));
              Real Gil_damp = PhysConst::mu0 * PhysConst::mag_gamma*0.25*(mag_alpha_arr(i,j,k)/mag_Ms_arr(i,j,k)
                                 +mag_alpha_arr(i+1,j,k)/mag_Ms_arr(i+1,j,k)
                                 +mag_alpha_arr(i,j+1,k)/mag_Ms_arr(i,j+1,k)
                                 +mag_alpha_arr(i+1,j+1,k)/mag_Ms_arr(i+1,j+1,k));
              // x component on z-faces of grid
              Mz(i, j, k, 0) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( Mz(i, j, k, 1) * Hz(i, j, k) - Mz(i, j, k, 2) * Hy_ztemp)
                + dt * Gil_damp * ( Mz(i, j, k, 1) * (Mz(i, j, k, 0) * Hy_ztemp - Mz(i, j, k, 1) * Hx(i, j, k))
                - Mz(i, j, k, 2) * ( Mz(i, j, k, 2) * Hx_ztemp - Mz(i, j, k, 0) * Hz(i, j, k)));

              // y component on z-faces of grid
              Mz(i, j, k, 1) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( Mz(i, j, k, 2) * Hx_ztemp - Mz(i, j, k, 0) * Hz(i, j, k))
                + dt * Gil_damp * ( Mz(i, j, k, 2) * (Mz(i, j, k, 1) * Hz(i, j, k) - Mz(i, j, k, 2) * Hy_ztemp)
                - Mz(i, j, k, 0) * ( Mz(i, j, k, 0) * Hy_ztemp - Mz(i, j, k, 1) * Hx_ztemp));

              // z component on z-faces of grid
              Mz(i, j, k, 2) += dt * (-PhysConst::mu0 * PhysConst::mag_gamma) * ( Mz(i, j, k, 0) * Hy_ztemp - Mz(i, j, k, 1) * Hx_ztemp)
                + dt * Gil_damp * ( Mz(i, j, k, 0) * ( Mz(i, j, k, 2) * Hx_ztemp - Mz(i, j, k, 0) * Hz(i, j, k))
                - Mz(i, j, k, 1) * ( Mz(i, j, k, 1) * Hz(i, j, k) - My(i, j, k, 2) * Hy_ztemp));
            }
            );
        }
    }

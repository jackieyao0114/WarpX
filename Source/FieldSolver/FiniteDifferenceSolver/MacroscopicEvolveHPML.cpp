/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Utils/WarpXAlgorithmSelection.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#ifdef WARPX_DIM_RZ
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H"
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "BoundaryConditions/PMLComponent.H"
#include <AMReX_Gpu.H>
#include <AMReX.H>
#include "Utils/CoarsenIO.H"

using namespace amrex;

#ifdef WARPX_MAG_LLG

/**
 * \brief Update the H field, over one timestep
 */
void FiniteDifferenceSolver::MacroscopicEvolveHPML (
    std::array< amrex::MultiFab*, 3 > Hfield,
    std::array< amrex::MultiFab*, 3 > const Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties,
    amrex::MultiFab* const mu_mf ) {

   // Select algorithm (The choice of algorithm is a runtime option,
   // but we compile code for each algorithm, using templates)
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(Hfield, Efield, dt, macroscopic_properties, mu_mf);
    amrex::Abort("PML are not implemented in cylindrical geometry.");
#else
    if (m_do_nodal) {

        MacroscopicEvolveHPMLCartesian <CartesianNodalAlgorithm> ( Hfield, Efield, dt, macroscopic_properties, mu_mf );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::Yee) {

        MacroscopicEvolveHPMLCartesian <CartesianYeeAlgorithm> ( Hfield, Efield, dt, macroscopic_properties, mu_mf );

    } else if (m_fdtd_algo == MaxwellSolverAlgo::CKC) {

        MacroscopicEvolveHPMLCartesian <CartesianCKCAlgorithm> ( Hfield, Efield, dt, macroscopic_properties, mu_mf );

    } else {
        amrex::Abort("Unknown algorithm");
    }
#endif
}


#ifndef WARPX_DIM_RZ

template<typename T_Algo>
void FiniteDifferenceSolver::MacroscopicEvolveHPMLCartesian (
    std::array< amrex::MultiFab*, 3 > Hfield,
    std::array< amrex::MultiFab*, 3 > const Efield,
    amrex::Real const dt,
    std::unique_ptr<MacroscopicProperties> const& macroscopic_properties,
    amrex::MultiFab* const mu_mf ) {

    amrex::GpuArray<int, 3> const& mu_stag  = macroscopic_properties->mu_IndexType;
    amrex::GpuArray<int, 3> const& Hx_stag  = macroscopic_properties->Hx_IndexType;
    amrex::GpuArray<int, 3> const& Hy_stag  = macroscopic_properties->Hy_IndexType;
    amrex::GpuArray<int, 3> const& Hz_stag  = macroscopic_properties->Hz_IndexType;
    amrex::GpuArray<int, 3> const& macro_cr = macroscopic_properties->macro_cr_ratio;

    // Loop through the grids, and over the tiles within each grid
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

    for ( MFIter mfi(*Hfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Hx = Hfield[0]->array(mfi);
        Array4<Real> const& Hy = Hfield[1]->array(mfi);
        Array4<Real> const& Hz = Hfield[2]->array(mfi);
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        int const n_coefs_x = m_stencil_coefs_x.size();
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        int const n_coefs_y = m_stencil_coefs_y.size();
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = m_stencil_coefs_z.size();

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Hfield[0]->ixType().ixType());
        Box const& tby  = mfi.tilebox(Hfield[1]->ixType().ixType());
        Box const& tbz  = mfi.tilebox(Hfield[2]->ixType().ixType());

        // mu_mf will be imported but will only be called at grids where Ms == 0
        Array4<Real> const& mu_arr = mu_mf->array(mfi);

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Real mu_arrx = CoarsenIO::Interp( mu_arr, mu_stag, Hx_stag, macro_cr, i, j, k, 0);

                Hx(i, j, k, PMLComp::xz) += 1._rt / mu_arrx * dt * (
                    T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, PMLComp::yx)
                  + T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, PMLComp::yy)
                  + T_Algo::UpwardDz(Ey, coefs_z, n_coefs_z, i, j, k, PMLComp::yz) );
                Hx(i, j, k, PMLComp::xy) -= 1._rt / mu_arrx * dt * (
                    T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, PMLComp::zx)
                  + T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, PMLComp::zy)
                  + T_Algo::UpwardDy(Ez, coefs_y, n_coefs_y, i, j, k, PMLComp::zz) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Real mu_arry = CoarsenIO::Interp( mu_arr, mu_stag, Hy_stag, macro_cr, i, j, k, 0);
                
                Hy(i, j, k, PMLComp::yx) += 1._rt / mu_arry * dt * (
                    T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, PMLComp::zx)
                  + T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, PMLComp::zy)
                  + T_Algo::UpwardDx(Ez, coefs_x, n_coefs_x, i, j, k, PMLComp::zz) );
                Hy(i, j, k, PMLComp::yz) -= 1._rt / mu_arry * dt * (
                    T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, PMLComp::xx)
                  + T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, PMLComp::xy)
                  + T_Algo::UpwardDz(Ex, coefs_z, n_coefs_z, i, j, k, PMLComp::xz) );
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                Real mu_arrz = CoarsenIO::Interp( mu_arr, mu_stag, Hz_stag, macro_cr, i, j, k, 0);

                Hz(i, j, k, PMLComp::zy) += 1._rt / mu_arrz * dt * (
                    T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, PMLComp::xx)
                  + T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, PMLComp::xy)
                  + T_Algo::UpwardDy(Ex, coefs_y, n_coefs_y, i, j, k, PMLComp::xz) );
                Hz(i, j, k, PMLComp::zx) -= 1._rt / mu_arrz * dt * (
                    T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, PMLComp::yx)
                  + T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, PMLComp::yy)
                  + T_Algo::UpwardDx(Ey, coefs_x, n_coefs_x, i, j, k, PMLComp::yz) );
            }

        );

    }

}

#endif // corresponds to ifndef WARPX_DIM_RZ

#endif // #ifdef WARPX_MAG_LLG

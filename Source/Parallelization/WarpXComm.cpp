/* Copyright 2019 Andrew Myers, Aurore Blelly, Axel Huebl
 * David Grote, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpXComm_K.H"
#include "WarpX.H"
#include "WarpXSumGuardCells.H"
#include "Utils/CoarsenMR.H"

#include <algorithm>
#include <cstdlib>
#include <memory>

using namespace amrex;

void
WarpX::ExchangeWithPmlB (int lev)
{
    if (do_pml && pml[lev]->ok()) {
        pml[lev]->ExchangeB({ Bfield_fp[lev][0].get(),
                              Bfield_fp[lev][1].get(),
                              Bfield_fp[lev][2].get() },
                            { Bfield_cp[lev][0].get(),
                              Bfield_cp[lev][1].get(),
                              Bfield_cp[lev][2].get() },
                              do_pml_in_domain);
    }
}

void
WarpX::ExchangeWithPmlE (int lev)
{
    if (do_pml && pml[lev]->ok()) {
        pml[lev]->ExchangeE({ Efield_fp[lev][0].get(),
                              Efield_fp[lev][1].get(),
                              Efield_fp[lev][2].get() },
                            { Efield_cp[lev][0].get(),
                              Efield_cp[lev][1].get(),
                              Efield_cp[lev][2].get() },
                              do_pml_in_domain);
    }
}

void
WarpX::ExchangeWithPmlF (int lev)
{
    if (do_pml && pml[lev]->ok()) {
        pml[lev]->ExchangeF(F_fp[lev].get(),
                            F_cp[lev].get(),
                            do_pml_in_domain);
    }
}

void
WarpX::UpdateAuxilaryData ()
{
    WARPX_PROFILE("WarpX::UpdateAuxilaryData()");

    if (Bfield_aux[0][0]->ixType() == Bfield_fp[0][0]->ixType()) {
        UpdateAuxilaryDataSameType();
    } else {
        UpdateAuxilaryDataStagToNodal();
    }
}

void
WarpX::UpdateAuxilaryDataStagToNodal ()
{
    // For level 0, we only need to do the average.
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*Bfield_aux[0][0]); mfi.isValid(); ++mfi)
    {
        Array4<Real> const& bx_aux = Bfield_aux[0][0]->array(mfi);
        Array4<Real> const& by_aux = Bfield_aux[0][1]->array(mfi);
        Array4<Real> const& bz_aux = Bfield_aux[0][2]->array(mfi);
        Array4<Real const> const& bx_fp = Bfield_fp[0][0]->const_array(mfi);
        Array4<Real const> const& by_fp = Bfield_fp[0][1]->const_array(mfi);
        Array4<Real const> const& bz_fp = Bfield_fp[0][2]->const_array(mfi);

        Array4<Real> const& ex_aux = Efield_aux[0][0]->array(mfi);
        Array4<Real> const& ey_aux = Efield_aux[0][1]->array(mfi);
        Array4<Real> const& ez_aux = Efield_aux[0][2]->array(mfi);
        Array4<Real const> const& ex_fp = Efield_fp[0][0]->const_array(mfi);
        Array4<Real const> const& ey_fp = Efield_fp[0][1]->const_array(mfi);
        Array4<Real const> const& ez_fp = Efield_fp[0][2]->const_array(mfi);

        const Box& bx = mfi.fabbox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
        {
            warpx_interp_nd_bfield_x(j,k,l, bx_aux, bx_fp);
            warpx_interp_nd_bfield_y(j,k,l, by_aux, by_fp);
            warpx_interp_nd_bfield_z(j,k,l, bz_aux, bz_fp);
            warpx_interp_nd_efield_x(j,k,l, ex_aux, ex_fp);
            warpx_interp_nd_efield_y(j,k,l, ey_aux, ey_fp);
            warpx_interp_nd_efield_z(j,k,l, ez_aux, ez_fp);
        });
    }

    for (int lev = 1; lev <= finest_level; ++lev)
    {
        BoxArray const& nba = Bfield_aux[lev][0]->boxArray();
        BoxArray const& cnba = amrex::coarsen(nba,2);
        DistributionMapping const& dm = Bfield_aux[lev][0]->DistributionMap();
        auto const& cperiod = Geom(lev-1).periodicity();

        // Bfield
        {
            Array<std::unique_ptr<MultiFab>,3> Btmp;
            if (Bfield_cax[lev][0]) {
                for (int i = 0; i < 3; ++i) {
                    Btmp[i] = std::make_unique<MultiFab>(
                        *Bfield_cax[lev][i], amrex::make_alias, 0, 1);
                }
            } else {
                IntVect ngtmp = Bfield_aux[lev-1][0]->nGrowVect();
                for (int i = 0; i < 3; ++i) {
                    Btmp[i] = std::make_unique<MultiFab>(cnba, dm, 1, ngtmp);
                }
            }
            // ParallelCopy from coarse level
            for (int i = 0; i < 3; ++i) {
                IntVect ng = Btmp[i]->nGrowVect();
                Btmp[i]->ParallelCopy(*Bfield_aux[lev-1][i], 0, 0, 1, ng, ng, cperiod);
            }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*Bfield_aux[lev][0]); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& bx_aux = Bfield_aux[lev][0]->array(mfi);
                Array4<Real> const& by_aux = Bfield_aux[lev][1]->array(mfi);
                Array4<Real> const& bz_aux = Bfield_aux[lev][2]->array(mfi);
                Array4<Real const> const& bx_fp = Bfield_fp[lev][0]->const_array(mfi);
                Array4<Real const> const& by_fp = Bfield_fp[lev][1]->const_array(mfi);
                Array4<Real const> const& bz_fp = Bfield_fp[lev][2]->const_array(mfi);
                Array4<Real const> const& bx_cp = Bfield_cp[lev][0]->const_array(mfi);
                Array4<Real const> const& by_cp = Bfield_cp[lev][1]->const_array(mfi);
                Array4<Real const> const& bz_cp = Bfield_cp[lev][2]->const_array(mfi);
                Array4<Real const> const& bx_c = Btmp[0]->const_array(mfi);
                Array4<Real const> const& by_c = Btmp[1]->const_array(mfi);
                Array4<Real const> const& bz_c = Btmp[2]->const_array(mfi);

                const Box& bx = mfi.fabbox();
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_nd_bfield_x(j,k,l, bx_aux, bx_fp, bx_cp, bx_c);
                    warpx_interp_nd_bfield_y(j,k,l, by_aux, by_fp, by_cp, by_c);
                    warpx_interp_nd_bfield_z(j,k,l, bz_aux, bz_fp, bz_cp, bz_c);
                });
            }
        }

        // Efield
        {
            Array<std::unique_ptr<MultiFab>,3> Etmp;
            if (Efield_cax[lev][0]) {
                for (int i = 0; i < 3; ++i) {
                    Etmp[i] = std::make_unique<MultiFab>(
                        *Efield_cax[lev][i], amrex::make_alias, 0, 1);
                }
            } else {
                IntVect ngtmp = Efield_aux[lev-1][0]->nGrowVect();
                for (int i = 0; i < 3; ++i) {
                    Etmp[i] = std::make_unique<MultiFab>(
                        cnba, dm, 1, ngtmp);
                }
            }
            // ParallelCopy from coarse level
            for (int i = 0; i < 3; ++i) {
                IntVect ng = Etmp[i]->nGrowVect();
                Etmp[i]->ParallelCopy(*Efield_aux[lev-1][i], 0, 0, 1, ng, ng, cperiod);
            }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*Efield_aux[lev][0]); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& ex_aux = Efield_aux[lev][0]->array(mfi);
                Array4<Real> const& ey_aux = Efield_aux[lev][1]->array(mfi);
                Array4<Real> const& ez_aux = Efield_aux[lev][2]->array(mfi);
                Array4<Real const> const& ex_fp = Efield_fp[lev][0]->const_array(mfi);
                Array4<Real const> const& ey_fp = Efield_fp[lev][1]->const_array(mfi);
                Array4<Real const> const& ez_fp = Efield_fp[lev][2]->const_array(mfi);
                Array4<Real const> const& ex_cp = Efield_cp[lev][0]->const_array(mfi);
                Array4<Real const> const& ey_cp = Efield_cp[lev][1]->const_array(mfi);
                Array4<Real const> const& ez_cp = Efield_cp[lev][2]->const_array(mfi);
                Array4<Real const> const& ex_c = Etmp[0]->const_array(mfi);
                Array4<Real const> const& ey_c = Etmp[1]->const_array(mfi);
                Array4<Real const> const& ez_c = Etmp[2]->const_array(mfi);

                const Box& bx = mfi.fabbox();
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_nd_efield_x(j,k,l, ex_aux, ex_fp, ex_cp, ex_c);
                    warpx_interp_nd_efield_y(j,k,l, ey_aux, ey_fp, ey_cp, ey_c);
                    warpx_interp_nd_efield_z(j,k,l, ez_aux, ez_fp, ez_cp, ez_c);
                });
            }
        }
    }
}

void
WarpX::UpdateAuxilaryDataSameType ()
{
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        const auto& crse_period = Geom(lev-1).periodicity();
        const IntVect& ng = Bfield_cp[lev][0]->nGrowVect();
        const DistributionMapping& dm = Bfield_cp[lev][0]->DistributionMap();

        // B field
        {
            MultiFab dBx(Bfield_cp[lev][0]->boxArray(), dm, Bfield_cp[lev][0]->nComp(), ng);
            MultiFab dBy(Bfield_cp[lev][1]->boxArray(), dm, Bfield_cp[lev][1]->nComp(), ng);
            MultiFab dBz(Bfield_cp[lev][2]->boxArray(), dm, Bfield_cp[lev][2]->nComp(), ng);
            dBx.setVal(0.0);
            dBy.setVal(0.0);
            dBz.setVal(0.0);
            dBx.ParallelCopy(*Bfield_aux[lev-1][0], 0, 0, Bfield_aux[lev-1][0]->nComp(), ng, ng, crse_period);
            dBy.ParallelCopy(*Bfield_aux[lev-1][1], 0, 0, Bfield_aux[lev-1][1]->nComp(), ng, ng, crse_period);
            dBz.ParallelCopy(*Bfield_aux[lev-1][2], 0, 0, Bfield_aux[lev-1][2]->nComp(), ng, ng, crse_period);
            if (Bfield_cax[lev][0])
            {
                MultiFab::Copy(*Bfield_cax[lev][0], dBx, 0, 0, Bfield_cax[lev][0]->nComp(), ng);
                MultiFab::Copy(*Bfield_cax[lev][1], dBy, 0, 0, Bfield_cax[lev][1]->nComp(), ng);
                MultiFab::Copy(*Bfield_cax[lev][2], dBz, 0, 0, Bfield_cax[lev][2]->nComp(), ng);
            }
            MultiFab::Subtract(dBx, *Bfield_cp[lev][0], 0, 0, Bfield_cp[lev][0]->nComp(), ng);
            MultiFab::Subtract(dBy, *Bfield_cp[lev][1], 0, 0, Bfield_cp[lev][1]->nComp(), ng);
            MultiFab::Subtract(dBz, *Bfield_cp[lev][2], 0, 0, Bfield_cp[lev][2]->nComp(), ng);

            const int refinement_ratio = refRatio(lev-1)[0];
            AMREX_ALWAYS_ASSERT(refinement_ratio == 2);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*Bfield_aux[lev][0]); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& bx_aux = Bfield_aux[lev][0]->array(mfi);
                Array4<Real> const& by_aux = Bfield_aux[lev][1]->array(mfi);
                Array4<Real> const& bz_aux = Bfield_aux[lev][2]->array(mfi);
                Array4<Real const> const& bx_fp = Bfield_fp[lev][0]->const_array(mfi);
                Array4<Real const> const& by_fp = Bfield_fp[lev][1]->const_array(mfi);
                Array4<Real const> const& bz_fp = Bfield_fp[lev][2]->const_array(mfi);
                Array4<Real const> const& bx_c = dBx.const_array(mfi);
                Array4<Real const> const& by_c = dBy.const_array(mfi);
                Array4<Real const> const& bz_c = dBz.const_array(mfi);

                amrex::ParallelFor(Box(bx_aux), Box(by_aux), Box(bz_aux),
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_bfield_x(j,k,l, bx_aux, bx_fp, bx_c);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_bfield_y(j,k,l, by_aux, by_fp, by_c);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_bfield_z(j,k,l, bz_aux, bz_fp, bz_c);
                });
            }
        }

        // E field
        {
            MultiFab dEx(Efield_cp[lev][0]->boxArray(), dm, Efield_cp[lev][0]->nComp(), ng);
            MultiFab dEy(Efield_cp[lev][1]->boxArray(), dm, Efield_cp[lev][1]->nComp(), ng);
            MultiFab dEz(Efield_cp[lev][2]->boxArray(), dm, Efield_cp[lev][2]->nComp(), ng);
            dEx.setVal(0.0);
            dEy.setVal(0.0);
            dEz.setVal(0.0);
            dEx.ParallelCopy(*Efield_aux[lev-1][0], 0, 0, Efield_aux[lev-1][0]->nComp(), ng, ng, crse_period);
            dEy.ParallelCopy(*Efield_aux[lev-1][1], 0, 0, Efield_aux[lev-1][1]->nComp(), ng, ng, crse_period);
            dEz.ParallelCopy(*Efield_aux[lev-1][2], 0, 0, Efield_aux[lev-1][2]->nComp(), ng, ng, crse_period);
            if (Efield_cax[lev][0])
            {
                MultiFab::Copy(*Efield_cax[lev][0], dEx, 0, 0, Efield_cax[lev][0]->nComp(), ng);
                MultiFab::Copy(*Efield_cax[lev][1], dEy, 0, 0, Efield_cax[lev][1]->nComp(), ng);
                MultiFab::Copy(*Efield_cax[lev][2], dEz, 0, 0, Efield_cax[lev][2]->nComp(), ng);
            }
            MultiFab::Subtract(dEx, *Efield_cp[lev][0], 0, 0, Efield_cp[lev][0]->nComp(), ng);
            MultiFab::Subtract(dEy, *Efield_cp[lev][1], 0, 0, Efield_cp[lev][1]->nComp(), ng);
            MultiFab::Subtract(dEz, *Efield_cp[lev][2], 0, 0, Efield_cp[lev][2]->nComp(), ng);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*Efield_aux[lev][0]); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& ex_aux = Efield_aux[lev][0]->array(mfi);
                Array4<Real> const& ey_aux = Efield_aux[lev][1]->array(mfi);
                Array4<Real> const& ez_aux = Efield_aux[lev][2]->array(mfi);
                Array4<Real const> const& ex_fp = Efield_fp[lev][0]->const_array(mfi);
                Array4<Real const> const& ey_fp = Efield_fp[lev][1]->const_array(mfi);
                Array4<Real const> const& ez_fp = Efield_fp[lev][2]->const_array(mfi);
                Array4<Real const> const& ex_c = dEx.const_array(mfi);
                Array4<Real const> const& ey_c = dEy.const_array(mfi);
                Array4<Real const> const& ez_c = dEz.const_array(mfi);

                amrex::ParallelFor(Box(ex_aux), Box(ey_aux), Box(ez_aux),
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_efield_x(j,k,l, ex_aux, ex_fp, ex_c);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_efield_y(j,k,l, ey_aux, ey_fp, ey_c);
                },
                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
                {
                    warpx_interp_efield_z(j,k,l, ez_aux, ez_fp, ez_c);
                });
            }
        }
    }
}

void
WarpX::FillBoundaryB (IntVect ng, IntVect ng_extra_fine)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryB(lev, ng, ng_extra_fine);
    }
}

void
WarpX::FillBoundaryE (IntVect ng, IntVect ng_extra_fine)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryE(lev, ng, ng_extra_fine);
    }
}

void
WarpX::FillBoundaryF (IntVect ng)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryF(lev, ng);
    }
}

void
WarpX::FillBoundaryB_avg (IntVect ng, IntVect ng_extra_fine)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryB_avg(lev, ng, ng_extra_fine);
    }
}

void
WarpX::FillBoundaryE_avg (IntVect ng, IntVect ng_extra_fine)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryE_avg(lev, ng, ng_extra_fine);
    }
}


void
WarpX::FillBoundaryE(int lev, IntVect ng, IntVect ng_extra_fine)
{
    FillBoundaryE(lev, PatchType::fine, ng+ng_extra_fine);
    if (lev > 0) FillBoundaryE(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryE (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->ExchangeE(patch_type,
                                { Efield_fp[lev][0].get(),
                                  Efield_fp[lev][1].get(),
                                  Efield_fp[lev][2].get() },
                                do_pml_in_domain);
            pml[lev]->FillBoundaryE(patch_type);
        }

        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ){
            Vector<MultiFab*> mf{Efield_fp[lev][0].get(),Efield_fp[lev][1].get(),Efield_fp[lev][2].get()};
            amrex::FillBoundary(mf, period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Efield_fp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryE, requested more guard cells than allocated");
            Efield_fp[lev][0]->FillBoundary(ng, period);
            Efield_fp[lev][1]->FillBoundary(ng, period);
            Efield_fp[lev][2]->FillBoundary(ng, period);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->ExchangeE(patch_type,
                                { Efield_cp[lev][0].get(),
                                  Efield_cp[lev][1].get(),
                                  Efield_cp[lev][2].get() },
                                do_pml_in_domain);
            pml[lev]->FillBoundaryE(patch_type);
        }
        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ) {
            Vector<MultiFab*> mf{Efield_cp[lev][0].get(),Efield_cp[lev][1].get(),Efield_cp[lev][2].get()};
            amrex::FillBoundary(mf, cperiod);

        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Efield_cp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryE, requested more guard cells than allocated");
            Efield_cp[lev][0]->FillBoundary(ng, cperiod);
            Efield_cp[lev][1]->FillBoundary(ng, cperiod);
            Efield_cp[lev][2]->FillBoundary(ng, cperiod);
        }
    }
}

void
WarpX::FillBoundaryB (int lev, IntVect ng, IntVect ng_extra_fine)
{
    FillBoundaryB(lev, PatchType::fine, ng + ng_extra_fine);
    if (lev > 0) FillBoundaryB(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryB (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->ExchangeB(patch_type,
                            { Bfield_fp[lev][0].get(),
                              Bfield_fp[lev][1].get(),
                              Bfield_fp[lev][2].get() },
                              do_pml_in_domain);
        pml[lev]->FillBoundaryB(patch_type);
        }
        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ) {
            Vector<MultiFab*> mf{Bfield_fp[lev][0].get(),Bfield_fp[lev][1].get(),Bfield_fp[lev][2].get()};
            amrex::FillBoundary(mf, period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Bfield_fp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryB, requested more guard cells than allocated");
            Bfield_fp[lev][0]->FillBoundary(ng, period);
            Bfield_fp[lev][1]->FillBoundary(ng, period);
            Bfield_fp[lev][2]->FillBoundary(ng, period);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev]->ok())
        {
        pml[lev]->ExchangeB(patch_type,
                      { Bfield_cp[lev][0].get(),
                        Bfield_cp[lev][1].get(),
                        Bfield_cp[lev][2].get() },
                        do_pml_in_domain);
        pml[lev]->FillBoundaryB(patch_type);
        }
        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ){
            Vector<MultiFab*> mf{Bfield_cp[lev][0].get(),Bfield_cp[lev][1].get(),Bfield_cp[lev][2].get()};
            amrex::FillBoundary(mf, cperiod);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Bfield_cp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryB, requested more guard cells than allocated");
            Bfield_cp[lev][0]->FillBoundary(ng, cperiod);
            Bfield_cp[lev][1]->FillBoundary(ng, cperiod);
            Bfield_cp[lev][2]->FillBoundary(ng, cperiod);
        }
    }
}

#ifdef WARPX_MAG_LLG
void
WarpX::FillBoundaryM (IntVect ng, IntVect ng_extra_fine)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryM(lev, ng, ng_extra_fine);
    }
}

void
WarpX::FillBoundaryM (int lev, IntVect ng, IntVect ng_extra_fine)
{
    FillBoundaryM(lev, PatchType::fine, ng + ng_extra_fine);
    if (lev > 0) FillBoundaryM(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryM (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
        {
            // ExchangeM not needed for PML algorithm
        }
        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ) {
            Vector<MultiFab*> mf{Mfield_fp[lev][0].get(),Mfield_fp[lev][1].get(),Mfield_fp[lev][2].get()};
            amrex::FillBoundary(mf, period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Mfield_fp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryM, requested more guard cells than allocated");
            Mfield_fp[lev][0]->FillBoundary(ng, period);
            Mfield_fp[lev][1]->FillBoundary(ng, period);
            Mfield_fp[lev][2]->FillBoundary(ng, period); //Mfield_fp is an amrex object
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        amrex::Abort("EvolveHM does not come with coarse patch yet");
        /*
        if (do_pml && pml[lev]->ok())
        {
        pml[lev]->ExchangeM(patch_type,
                      { Mfield_cp[lev][0].get(),
                        Mfield_cp[lev][1].get(),
                        Mfield_cp[lev][2].get() },
                        do_pml_in_domain);
        pml[lev]->FillBoundaryM(patch_type);
        }
        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ){
            Vector<MultiFab*> mf{Mfield_cp[lev][0].get(),Mfield_cp[lev][1].get(),Mfield_cp[lev][2].get()};
            amrex::FillBoundary(mf, cperiod);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Mfield_cp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryM, requested more guard cells than allocated");
            Mfield_cp[lev][0]->FillBoundary(ng, cperiod);
            Mfield_cp[lev][1]->FillBoundary(ng, cperiod);
            Mfield_cp[lev][2]->FillBoundary(ng, cperiod);
        }
        */
    }
}

void
WarpX::FillBoundaryH (IntVect ng, IntVect ng_extra_fine)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        FillBoundaryH(lev, ng, ng_extra_fine);
    }
}

void
WarpX::FillBoundaryH (int lev, IntVect ng, IntVect ng_extra_fine)
{
    FillBoundaryH(lev, PatchType::fine, ng + ng_extra_fine);
    if (lev > 0) FillBoundaryH(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryH (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->ExchangeH(patch_type,
                            { Hfield_fp[lev][0].get(),
                              Hfield_fp[lev][1].get(),
                              Hfield_fp[lev][2].get() },
                              do_pml_in_domain);
            pml[lev]->FillBoundaryH(patch_type);
        }
        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ) {
            Vector<MultiFab*> mf{Hfield_fp[lev][0].get(),Hfield_fp[lev][1].get(),Hfield_fp[lev][2].get()};
            amrex::FillBoundary(mf, period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Hfield_fp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryH, requested more guard cells than allocated");
            Hfield_fp[lev][0]->FillBoundary(ng, period);
            Hfield_fp[lev][1]->FillBoundary(ng, period);
            Hfield_fp[lev][2]->FillBoundary(ng, period); //Hfield_fp is an amrex object
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        amrex::Abort("EvolveHM does not come with coarse patch yet");
        /*
        if (do_pml && pml[lev]->ok())
        {
        pml[lev]->ExchangeH(patch_type,
                      { Hfield_cp[lev][0].get(),
                        Hfield_cp[lev][1].get(),
                        Hfield_cp[lev][2].get() },
                        do_pml_in_domain);
        pml[lev]->FillBoundaryH(patch_type);
        }
        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ){
            Vector<MultiFab*> mf{Hfield_cp[lev][0].get(),Hfield_cp[lev][1].get(),Hfield_cp[lev][2].get()};
            amrex::FillBoundary(mf, cperiod);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Hfield_cp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryH, requested more guard cells than allocated");
            Hfield_cp[lev][0]->FillBoundary(ng, cperiod);
            Hfield_cp[lev][1]->FillBoundary(ng, cperiod);
            Hfield_cp[lev][2]->FillBoundary(ng, cperiod);
        }
        */
    }
}
#endif

void
WarpX::FillBoundaryE_avg(int lev, IntVect ng, IntVect ng_extra_fine)
{
    FillBoundaryE_avg(lev, PatchType::fine, ng+ng_extra_fine);
    if (lev > 0) FillBoundaryE_avg(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryE_avg (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
         {
            amrex::Abort("Averaged Galilean PSATD with PML is not yet implemented");
         }

        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ){
            Vector<MultiFab*> mf{Efield_avg_fp[lev][0].get(),Efield_avg_fp[lev][1].get(),Efield_avg_fp[lev][2].get()};
            amrex::FillBoundary(mf, period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Efield_avg_fp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryE_avg, requested more guard cells than allocated");
            Efield_avg_fp[lev][0]->FillBoundary(ng, period);
            Efield_avg_fp[lev][1]->FillBoundary(ng, period);
            Efield_avg_fp[lev][2]->FillBoundary(ng, period);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev]->ok())
         {
            amrex::Abort("Averaged Galilean PSATD with PML is not yet implemented");
         }

        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ) {
            Vector<MultiFab*> mf{Efield_avg_cp[lev][0].get(),Efield_avg_cp[lev][1].get(),Efield_avg_cp[lev][2].get()};
            amrex::FillBoundary(mf, cperiod);

        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Efield_avg_cp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryE, requested more guard cells than allocated");
            Efield_avg_cp[lev][0]->FillBoundary(ng, cperiod);
            Efield_avg_cp[lev][1]->FillBoundary(ng, cperiod);
            Efield_avg_cp[lev][2]->FillBoundary(ng, cperiod);
        }
    }
}


void
WarpX::FillBoundaryB_avg (int lev, IntVect ng, IntVect ng_extra_fine)
{
    FillBoundaryB_avg(lev, PatchType::fine, ng + ng_extra_fine);
    if (lev > 0) FillBoundaryB_avg(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryB_avg (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine)
    {
        if (do_pml && pml[lev]->ok())
          {
            amrex::Abort("Averaged Galilean PSATD with PML is not yet implemented");
          }
        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ) {
            Vector<MultiFab*> mf{Bfield_avg_fp[lev][0].get(),Bfield_avg_fp[lev][1].get(),Bfield_avg_fp[lev][2].get()};
            amrex::FillBoundary(mf, period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Bfield_fp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryB, requested more guard cells than allocated");
            Bfield_avg_fp[lev][0]->FillBoundary(ng, period);
            Bfield_avg_fp[lev][1]->FillBoundary(ng, period);
            Bfield_avg_fp[lev][2]->FillBoundary(ng, period);
        }
    }
    else if (patch_type == PatchType::coarse)
    {
        if (do_pml && pml[lev]->ok())
          {
            amrex::Abort("Averaged Galilean PSATD with PML is not yet implemented");
          }

        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ){
            Vector<MultiFab*> mf{Bfield_avg_cp[lev][0].get(),Bfield_avg_cp[lev][1].get(),Bfield_avg_cp[lev][2].get()};
            amrex::FillBoundary(mf, cperiod);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= Bfield_avg_cp[lev][0]->nGrowVect(),
                "Error: in FillBoundaryB_avg, requested more guard cells than allocated");
            Bfield_avg_cp[lev][0]->FillBoundary(ng, cperiod);
            Bfield_avg_cp[lev][1]->FillBoundary(ng, cperiod);
            Bfield_avg_cp[lev][2]->FillBoundary(ng, cperiod);
        }
    }
}


void
WarpX::FillBoundaryF (int lev, IntVect ng)
{
    FillBoundaryF(lev, PatchType::fine, ng);
    if (lev > 0) FillBoundaryF(lev, PatchType::coarse, ng);
}

void
WarpX::FillBoundaryF (int lev, PatchType patch_type, IntVect ng)
{
    if (patch_type == PatchType::fine && F_fp[lev])
    {
        if (do_pml && pml[lev]->ok())
        {
            pml[lev]->ExchangeF(patch_type, F_fp[lev].get(),
                                do_pml_in_domain);
            pml[lev]->FillBoundaryF(patch_type);
        }

        const auto& period = Geom(lev).periodicity();
        if ( safe_guard_cells ) {
            F_fp[lev]->FillBoundary(period);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= F_fp[lev]->nGrowVect(),
                "Error: in FillBoundaryF, requested more guard cells than allocated");
            F_fp[lev]->FillBoundary(ng, period);
        }
    }
    else if (patch_type == PatchType::coarse && F_cp[lev])
    {
        if (do_pml && pml[lev]->ok())
        {
        pml[lev]->ExchangeF(patch_type, F_cp[lev].get(),
                            do_pml_in_domain);
        pml[lev]->FillBoundaryF(patch_type);
        }

        const auto& cperiod = Geom(lev-1).periodicity();
        if ( safe_guard_cells ) {
            F_cp[lev]->FillBoundary(cperiod);
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                ng <= F_cp[lev]->nGrowVect(),
                "Error: in FillBoundaryF, requested more guard cells than allocated");
            F_cp[lev]->FillBoundary(ng, cperiod);
        }
    }
}

void
WarpX::FillBoundaryAux (IntVect ng)
{
    for (int lev = 0; lev <= finest_level-1; ++lev)
    {
        FillBoundaryAux(lev, ng);
    }
}

void
WarpX::FillBoundaryAux (int lev, IntVect ng)
{
    const auto& period = Geom(lev).periodicity();
    Efield_aux[lev][0]->FillBoundary(ng, period);
    Efield_aux[lev][1]->FillBoundary(ng, period);
    Efield_aux[lev][2]->FillBoundary(ng, period);
    Bfield_aux[lev][0]->FillBoundary(ng, period);
    Bfield_aux[lev][1]->FillBoundary(ng, period);
    Bfield_aux[lev][2]->FillBoundary(ng, period);
#ifdef WARPX_MAG_LLG
    Mfield_aux[lev][0]->FillBoundary(ng, period);
    Mfield_aux[lev][1]->FillBoundary(ng, period);
    Mfield_aux[lev][2]->FillBoundary(ng, period);
#endif
}

void
WarpX::SyncCurrent ()
{
    WARPX_PROFILE("WarpX::SyncCurrent()");

    // Restrict fine patch current onto the coarse patch, before
    // summing the guard cells of the fine patch
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        current_cp[lev][0]->setVal(0.0);
        current_cp[lev][1]->setVal(0.0);
        current_cp[lev][2]->setVal(0.0);

        const IntVect& refinement_ratio = refRatio(lev-1);

        std::array<const MultiFab*,3> fine { current_fp[lev][0].get(),
                                             current_fp[lev][1].get(),
                                             current_fp[lev][2].get() };
        std::array<      MultiFab*,3> crse { current_cp[lev][0].get(),
                                             current_cp[lev][1].get(),
                                             current_cp[lev][2].get() };
        CoarsenMR::Coarsen( *crse[0], *fine[0], refinement_ratio );
        CoarsenMR::Coarsen( *crse[1], *fine[1], refinement_ratio );
        CoarsenMR::Coarsen( *crse[2], *fine[2], refinement_ratio );
    }

    // For each level
    // - apply filter to the coarse patch/buffer of `lev+1` and fine patch of `lev` (same resolution)
    // - add the coarse patch/buffer of `lev+1` into the fine patch of `lev`
    // - sum guard cells of the coarse patch of `lev+1` and fine patch of `lev`
    for (int lev=0; lev <= finest_level; ++lev) {
        AddCurrentFromFineLevelandSumBoundary(lev);
    }
}

void
WarpX::SyncRho ()
{
    WARPX_PROFILE("WarpX::SyncRho()");

    if (!rho_fp[0]) return;
    const int ncomp = rho_fp[0]->nComp();

    // Restrict fine patch onto the coarse patch,
    // before summing the guard cells of the fine patch
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        rho_cp[lev]->setVal(0.0);
        const IntVect& refinement_ratio = refRatio(lev-1);
        CoarsenMR::Coarsen( *rho_cp[lev], *rho_fp[lev], refinement_ratio );
    }

    // For each level
    // - apply filter to the coarse patch/buffer of `lev+1` and fine patch of `lev` (same resolution)
    // - add the coarse patch/buffer of `lev+1` into the fine patch of `lev`
    // - sum guard cells of the coarse patch of `lev+1` and fine patch of `lev`
    for (int lev=0; lev <= finest_level; ++lev) {
        AddRhoFromFineLevelandSumBoundary(lev, 0, ncomp);
    }
}

/** \brief Fills the values of the current on the coarse patch by
 *  averaging the values of the current of the fine patch (on the same level).
 */
void
WarpX::RestrictCurrentFromFineToCoarsePatch (int lev)
{
    current_cp[lev][0]->setVal(0.0);
    current_cp[lev][1]->setVal(0.0);
    current_cp[lev][2]->setVal(0.0);

    const IntVect& refinement_ratio = refRatio(lev-1);

    std::array<const MultiFab*,3> fine { current_fp[lev][0].get(),
                                         current_fp[lev][1].get(),
                                         current_fp[lev][2].get() };
    std::array<      MultiFab*,3> crse { current_cp[lev][0].get(),
                                         current_cp[lev][1].get(),
                                         current_cp[lev][2].get() };
    CoarsenMR::Coarsen( *crse[0], *fine[0], refinement_ratio );
    CoarsenMR::Coarsen( *crse[1], *fine[1], refinement_ratio );
    CoarsenMR::Coarsen( *crse[2], *fine[2], refinement_ratio );
}

void
WarpX::ApplyFilterandSumBoundaryJ (int lev, PatchType patch_type)
{
    const int glev = (patch_type == PatchType::fine) ? lev : lev-1;
    const auto& period = Geom(glev).periodicity();
    auto& j = (patch_type == PatchType::fine) ? current_fp[lev] : current_cp[lev];
    for (int idim = 0; idim < 3; ++idim) {
        if (use_filter) {
            IntVect ng = j[idim]->nGrowVect();
            ng += bilinear_filter.stencil_length_each_dir-1;
            MultiFab jf(j[idim]->boxArray(), j[idim]->DistributionMap(), j[idim]->nComp(), ng);
            bilinear_filter.ApplyStencil(jf, *j[idim]);
            WarpXSumGuardCells(*(j[idim]), jf, period, 0, (j[idim])->nComp());
        } else {
            WarpXSumGuardCells(*(j[idim]), period, 0, (j[idim])->nComp());
        }
    }
}

/* /brief Update the currents of `lev` by adding the currents from particles
*         that are in the mesh refinement patches at `lev+1`
*
* More precisely, apply filter and sum boundaries for the current of:
* - the fine patch of `lev`
* - the coarse patch of `lev+1` (same resolution)
* - the buffer regions of the coarse patch of `lev+1` (i.e. for particules
* that are within the mesh refinement patch, but do not deposit on the
* mesh refinement patch because they are too close to the boundary)
*
* Then update the fine patch of `lev` by adding the currents for the coarse
* patch (and buffer region) of `lev+1`
*/
void
WarpX::AddCurrentFromFineLevelandSumBoundary (int lev)
{
    ApplyFilterandSumBoundaryJ(lev, PatchType::fine);

    if (lev < finest_level) {
        // When there are current buffers, unlike coarse patch,
        // we don't care about the final state of them.

        const auto& period = Geom(lev).periodicity();
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab mf(current_fp[lev][idim]->boxArray(),
                        current_fp[lev][idim]->DistributionMap(), current_fp[lev][idim]->nComp(), 0);
            mf.setVal(0.0);
            if (use_filter && current_buf[lev+1][idim])
            {
                // coarse patch of fine level
                IntVect ng = current_cp[lev+1][idim]->nGrowVect();
                ng += bilinear_filter.stencil_length_each_dir-1;
                MultiFab jfc(current_cp[lev+1][idim]->boxArray(),
                             current_cp[lev+1][idim]->DistributionMap(), current_cp[lev+1][idim]->nComp(), ng);
                bilinear_filter.ApplyStencil(jfc, *current_cp[lev+1][idim]);

                // buffer patch of fine level
                MultiFab jfb(current_buf[lev+1][idim]->boxArray(),
                             current_buf[lev+1][idim]->DistributionMap(), current_buf[lev+1][idim]->nComp(), ng);
                bilinear_filter.ApplyStencil(jfb, *current_buf[lev+1][idim]);

                MultiFab::Add(jfb, jfc, 0, 0, current_buf[lev+1][idim]->nComp(), ng);
                mf.ParallelAdd(jfb, 0, 0, current_buf[lev+1][idim]->nComp(), ng, IntVect::TheZeroVector(), period);

                WarpXSumGuardCells(*current_cp[lev+1][idim], jfc, period, 0, current_cp[lev+1][idim]->nComp());
            }
            else if (use_filter) // but no buffer
            {
                // coarse patch of fine level
                IntVect ng = current_cp[lev+1][idim]->nGrowVect();
                ng += bilinear_filter.stencil_length_each_dir-1;
                MultiFab jf(current_cp[lev+1][idim]->boxArray(),
                            current_cp[lev+1][idim]->DistributionMap(), current_cp[lev+1][idim]->nComp(), ng);
                bilinear_filter.ApplyStencil(jf, *current_cp[lev+1][idim]);
                mf.ParallelAdd(jf, 0, 0, current_cp[lev+1][idim]->nComp(), ng, IntVect::TheZeroVector(), period);
                WarpXSumGuardCells(*current_cp[lev+1][idim], jf, period, 0, current_cp[lev+1][idim]->nComp());
            }
            else if (current_buf[lev+1][idim]) // but no filter
            {
                MultiFab::Add(*current_buf[lev+1][idim],
                               *current_cp [lev+1][idim], 0, 0, current_buf[lev+1][idim]->nComp(),
                               current_cp[lev+1][idim]->nGrow());
                mf.ParallelAdd(*current_buf[lev+1][idim], 0, 0, current_buf[lev+1][idim]->nComp(),
                               current_buf[lev+1][idim]->nGrowVect(), IntVect::TheZeroVector(),
                               period);
                WarpXSumGuardCells(*(current_cp[lev+1][idim]), period, 0, current_cp[lev+1][idim]->nComp());
            }
            else // no filter, no buffer
            {
                mf.ParallelAdd(*current_cp[lev+1][idim], 0, 0, current_cp[lev+1][idim]->nComp(),
                               current_cp[lev+1][idim]->nGrowVect(), IntVect::TheZeroVector(),
                               period);
                WarpXSumGuardCells(*(current_cp[lev+1][idim]), period, 0, current_cp[lev+1][idim]->nComp());
            }
            MultiFab::Add(*current_fp[lev][idim], mf, 0, 0, current_fp[lev+1][idim]->nComp(), 0);
        }
        NodalSyncJ(lev+1, PatchType::coarse);
    }
    NodalSyncJ(lev, PatchType::fine);
}

void
WarpX::RestrictRhoFromFineToCoarsePatch (int lev)
{
    if (rho_fp[lev]) {
        rho_cp[lev]->setVal(0.0);
        const IntVect& refinement_ratio = refRatio(lev-1);
        CoarsenMR::Coarsen( *rho_cp[lev], *rho_fp[lev], refinement_ratio );
    }
}

void
WarpX::ApplyFilterandSumBoundaryRho (int lev, PatchType patch_type, int icomp, int ncomp)
{
    const int glev = (patch_type == PatchType::fine) ? lev : lev-1;
    auto& r = (patch_type == PatchType::fine) ? rho_fp[lev] : rho_cp[lev];
    if (r == nullptr) return;
    ApplyFilterandSumBoundaryRho(lev, glev, *r, icomp, ncomp);
}

void
WarpX::ApplyFilterandSumBoundaryRho (int /*lev*/, int glev, amrex::MultiFab& rho, int icomp, int ncomp)
{
    const auto& period = Geom(glev).periodicity();
    if (use_filter) {
        IntVect ng = rho.nGrowVect();
        ng += bilinear_filter.stencil_length_each_dir-1;
        MultiFab rf(rho.boxArray(), rho.DistributionMap(), ncomp, ng);
        bilinear_filter.ApplyStencil(rf, rho, icomp, 0, ncomp);
        WarpXSumGuardCells(rho, rf, period, icomp, ncomp );
    } else {
        WarpXSumGuardCells(rho, period, icomp, ncomp);
    }
}

/* /brief Update the charge density of `lev` by adding the charge density from particles
*         that are in the mesh refinement patches at `lev+1`
*
* More precisely, apply filter and sum boundaries for the charge density of:
* - the fine patch of `lev`
* - the coarse patch of `lev+1` (same resolution)
* - the buffer regions of the coarse patch of `lev+1` (i.e. for particules
* that are within the mesh refinement patch, but do not deposit on the
* mesh refinement patch because they are too close to the boundary)
*
* Then update the fine patch of `lev` by adding the charge density for the coarse
* patch (and buffer region) of `lev+1`
*/
void
WarpX::AddRhoFromFineLevelandSumBoundary(int lev, int icomp, int ncomp)
{
    if (!rho_fp[lev]) return;

    ApplyFilterandSumBoundaryRho(lev, PatchType::fine, icomp, ncomp);

    if (lev < finest_level){

        const auto& period = Geom(lev).periodicity();
        MultiFab mf(rho_fp[lev]->boxArray(),
                    rho_fp[lev]->DistributionMap(),
                    ncomp, 0);
        mf.setVal(0.0);
        if (use_filter && charge_buf[lev+1])
        {
            // coarse patch of fine level
            IntVect ng = rho_cp[lev+1]->nGrowVect();
            ng += bilinear_filter.stencil_length_each_dir-1;
            MultiFab rhofc(rho_cp[lev+1]->boxArray(),
                         rho_cp[lev+1]->DistributionMap(), ncomp, ng);
            bilinear_filter.ApplyStencil(rhofc, *rho_cp[lev+1], icomp, 0, ncomp);

            // buffer patch of fine level
            MultiFab rhofb(charge_buf[lev+1]->boxArray(),
                           charge_buf[lev+1]->DistributionMap(), ncomp, ng);
            bilinear_filter.ApplyStencil(rhofb, *charge_buf[lev+1], icomp, 0, ncomp);

            MultiFab::Add(rhofb, rhofc, 0, 0, ncomp, ng);
            mf.ParallelAdd(rhofb, 0, 0, ncomp, ng, IntVect::TheZeroVector(), period);
            WarpXSumGuardCells( *rho_cp[lev+1], rhofc, period, icomp, ncomp );
        }
        else if (use_filter) // but no buffer
        {
            IntVect ng = rho_cp[lev+1]->nGrowVect();
            ng += bilinear_filter.stencil_length_each_dir-1;
            MultiFab rf(rho_cp[lev+1]->boxArray(), rho_cp[lev+1]->DistributionMap(), ncomp, ng);
            bilinear_filter.ApplyStencil(rf, *rho_cp[lev+1], icomp, 0, ncomp);
            mf.ParallelAdd(rf, 0, 0, ncomp, ng, IntVect::TheZeroVector(), period);
            WarpXSumGuardCells( *rho_cp[lev+1], rf, period, icomp, ncomp );
        }
        else if (charge_buf[lev+1]) // but no filter
        {
            MultiFab::Add(*charge_buf[lev+1],
                           *rho_cp[lev+1], icomp, icomp, ncomp,
                           rho_cp[lev+1]->nGrow());
            mf.ParallelAdd(*charge_buf[lev+1], icomp, 0,
                           ncomp,
                           charge_buf[lev+1]->nGrowVect(), IntVect::TheZeroVector(),
                           period);
            WarpXSumGuardCells(*(rho_cp[lev+1]), period, icomp, ncomp);
        }
        else // no filter, no buffer
        {
            mf.ParallelAdd(*rho_cp[lev+1], icomp, 0, ncomp,
                           rho_cp[lev+1]->nGrowVect(), IntVect::TheZeroVector(),
                           period);
            WarpXSumGuardCells(*(rho_cp[lev+1]), period, icomp, ncomp);
        }
        MultiFab::Add(*rho_fp[lev], mf, 0, icomp, ncomp, 0);
        NodalSyncRho(lev+1, PatchType::coarse, icomp, ncomp);
    }

    NodalSyncRho(lev, PatchType::fine, icomp, ncomp);
}

void
WarpX::NodalSyncJ (int lev, PatchType patch_type)
{
    if (!override_sync_intervals.contains(istep[0])) return;

    if (patch_type == PatchType::fine)
    {
        const auto& period = Geom(lev).periodicity();
        current_fp[lev][0]->OverrideSync(period);
        current_fp[lev][1]->OverrideSync(period);
        current_fp[lev][2]->OverrideSync(period);
    }
    else if (patch_type == PatchType::coarse)
    {
        const auto& cperiod = Geom(lev-1).periodicity();
        current_cp[lev][0]->OverrideSync(cperiod);
        current_cp[lev][1]->OverrideSync(cperiod);
        current_cp[lev][2]->OverrideSync(cperiod);
    }
}

void
WarpX::NodalSyncRho (int lev, PatchType patch_type, int icomp, int ncomp)
{
    if (!override_sync_intervals.contains(istep[0])) return;

    if (patch_type == PatchType::fine && rho_fp[lev])
    {
        const auto& period = Geom(lev).periodicity();
        MultiFab rhof(*rho_fp[lev], amrex::make_alias, icomp, ncomp);
        rhof.OverrideSync(period);
    }
    else if (patch_type == PatchType::coarse && rho_cp[lev])
    {
        const auto& cperiod = Geom(lev-1).periodicity();
        MultiFab rhoc(*rho_cp[lev], amrex::make_alias, icomp, ncomp);
        rhoc.OverrideSync(cperiod);
    }
}

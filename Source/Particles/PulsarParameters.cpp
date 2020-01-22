#include <PulsarParameters.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

namespace PulsarParm
{
   std::string pulsar_type;

   AMREX_GPU_DEVICE_MANAGED amrex::Real omega_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real B_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real R_star;
   AMREX_GPU_DEVICE_MANAGED amrex::Real dR_star;
   AMREX_GPU_DEVICE_MANAGED int EB_external = 0;
   AMREX_GPU_DEVICE_MANAGED int E_external_monopole = 0;
   AMREX_GPU_DEVICE_MANAGED 
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center_star;
   AMREX_GPU_DEVICE_MANAGED int verbose = 0;

   void ReadParameters() {
      amrex::ParmParse pp("pulsar");
      pp.query("pulsarType",pulsar_type);
      pp.get("omega_star",omega_star);
      amrex::Vector<amrex::Real> center_star_v(AMREX_SPACEDIM);
      pp.getarr("center_star",center_star_v);
      std::copy(center_star_v.begin(),center_star_v.end(),center_star.begin());
      pp.get("R_star",R_star);
      pp.get("B_star",B_star);
      pp.get("dR",dR_star);
      pp.query("verbose",verbose);
      pp.query("EB_external",EB_external);
      pp.query("E_external_monopole",E_external_monopole);
      amrex::Print() << " Pulsar center: " << center_star[0] << " " << center_star[1] << " " << center_star[2] << "\n";
      amrex::Print() << " Pulsar omega: " << omega_star << "\n";
      amrex::Print() << " Pulsar B_star : " << B_star << "\n";
   }

}

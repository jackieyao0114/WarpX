#ifndef WARPX_EXTERNALFORCE_PM_H
#define WARPX_EXTERNALFORCE_PM_H

#include <AMReX_REAL.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void ApplyExternalForce(
    const amrex::Real xp, const amrex::Real yp, const amrex::Real zp, 
    const amrex::Real ux, const amrex::Real uy, const amrex::Real uz,
    amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
    amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz,
    const amrex::Real xc, const amrex::Real yc, const amrex::Real zc, 
    const amrex::Real q, const amrex::Real m, amrex::Real dt , const int istep, const amrex::Real wp)
{

    // spherical radius, theta, phi and cylindrical r //
    const amrex::Real r = std::sqrt( (xp-xc)*(xp-xc) + (yp-yc)*(yp-yc) + (zp-zc)*(zp-zc));
    const amrex::Real phi = std::atan2((yp-yc),(xp-xc));
    amrex::Real theta = 0.0;
    if (r>0) {
       theta = std::acos( (zp-zc)/r );
    }
    const amrex::Real r_cl = std::sin(theta) * r;
    
    // angular vel and radius of star 
    const amrex::Real omega_star = 1125000. ; // rotation axis is k_hat
    const amrex::Real r_star = 100.;
  
    // 
    const amrex::Real f_drag_coeff = -1.0 * m /(q*dt*50.0);    
    amrex::Real f_drag_ux, f_drag_uy, f_drag_uz;
    amrex::Real B_star, c_theta, s_theta, c_phi, s_phi;
    B_star = 1.0e8;
    c_theta = std::cos(theta);
    s_theta = std::sin(theta);
    c_phi = std::cos(phi);
    s_phi = std::sin(phi);
    // apply drag force only for particles inside the pulsar  
    if (r <= r_star*1.001)
    {

        constexpr amrex::Real inv_c2 = 1./(PhysConst::c*PhysConst::c);

        // Compute inverse Lorentz factor
        const amrex::Real inv_gamma = 1./std::sqrt(1. + (ux*ux + uy*uy + uz*uz)*inv_c2);

        f_drag_ux = ux*inv_gamma - (-1.0)*(r_cl/r_star)*r_star*omega_star*std::sin(phi);
        f_drag_uy = uy*inv_gamma - (r_cl/r_star)*r_star*omega_star*std::cos(phi);
        f_drag_uz = uz*inv_gamma - 0.0;
 
        Ex += f_drag_coeff * f_drag_ux ;
        Ey += f_drag_coeff * f_drag_uy ;
//        Ez += f_drag_coeff * f_drag_uz;
//        Ex -= (-1.0)*B_star*2.0*r_cl*omega_star*c_phi;
//        Ey -= (-1.0)*B_star*2.0*r_cl*omega_star*s_phi;

    }
    if(r <= r_star*1.001)
    {
           Bx += 0.0;
           By += 0.0;
           Bz += B_star*2.0; 
    }
    // External E and B field that correspond to a magnetic dipole //
    if(r > (r_star*1.001))
    {

           amrex::Real r_ratio = r_star/r;
           amrex::Real r3 = r_ratio*r_ratio*r_ratio;
           amrex::Real B_r = 2.0*B_star*r3*c_theta;
           amrex::Real B_theta = B_star*r3*s_theta;
           Bx += B_r*c_phi*s_theta + B_theta*c_phi*c_theta;
           By += B_r*s_phi*s_theta + B_theta*s_phi*c_theta;
           Bz += B_r*c_theta - B_theta*s_theta;
           amrex::Real E_r = omega_star*r_star*B_star*r3*r_ratio*(1.0-3.0*c_theta*c_theta) ;
//           amrex::Real E_r = 0.0 ;
//                             + (2.0/3.0)*omega_star*B_star*r_star*r_ratio*r_ratio;
           amrex::Real E_theta = (-1.0)*omega_star*B_star*r_star*r3*r_ratio*(c_theta*s_theta*2.0);
           Ex += E_r*c_phi*s_theta + E_theta*c_phi*c_theta;
           Ey += E_r*s_phi*s_theta + E_theta*s_phi*c_theta;
           Ez += 1.0*E_r*c_theta - (1.0)*E_theta*s_theta;
    }

}

#endif

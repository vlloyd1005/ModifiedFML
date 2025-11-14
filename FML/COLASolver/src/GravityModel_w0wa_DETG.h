#ifndef GRAVITYMODEL_W0WA_DETG_HEADER
#define GRAVITYMODEL_W0WA_DETG_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/MultigridSolver/DGPSolver.h>


/// DETG model
template <int NDIM>
class GravityModelDETG final : public GravityModel<NDIM> {
  protected:
    // For calculating the modification to the growth factor
    // Constant wrt time (a) but not class instance, initialized here
    double beta;
    double p_val;
    double OmegaDE0;
    

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    GravityModelDETG() : GravityModel<NDIM>("DETG") {}
    GravityModelDETG(std::shared_ptr<Cosmology> cosmo)
        : GravityModel<NDIM>(cosmo, "DETG"),
          beta(cosmo->get_beta()),
          p_val(cosmo->get_p()),
          OmegaDE0(cosmo->get_OmegaLambda())
    {}
    
    //========================================================================
    // Find modification factor
    //========================================================================
    double get_sqrt_alpha(double a) const {
        const double OmegaDEa = this->cosmo->get_OmegaLambda(a);
        double sqrt_alpha = std::sqrt((1 - beta * pow(OmegaDEa/OmegaDE0, p_val)));

        return sqrt_alpha;
    }
    
    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# Model (DETG) parameters:\n";
            std::cout << "# beta             : " << beta << "\n";
            std::cout << "# p_val            : " << p_val << "\n";
            std::cout << "# OmegaDE0         : " << OmegaDE0 << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Modify growth factor in 1LPT
    //========================================================================
    double get_D_1LPT(double a, double koverH0 = 0.0) const override {
        koverH0 = std::max(koverH0, this->koverH0low);
        double sqrt_alpha = get_sqrt_alpha(a);

        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "# DEBUG: sqrt_alpha is [" << sqrt_alpha << "]  at z = " << 1.0 / a - 1.0 << "\n";
        }
        
        return not this->scaledependent_growth ? this->D_1LPT_of_loga(std::log(a)) * sqrt_alpha :
                                                 this->D_1LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a)) * sqrt_alpha;
    }

    //========================================================================
    // In GR GeffOverG == 1
    //========================================================================
    double GeffOverG([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const override { return 1.0; }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi = norm_poisson_equation * delta
    //========================================================================
    void compute_force(double a,
                       [[maybe_unused]] double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Computes gravitational force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        FML::NBODY::compute_force_from_density_fourier<NDIM>(
            density_fourier, force_real, density_assignment_method_used, norm_poisson_equation);
    }

    
    //========================================================================
    // In GR GeffOverG == 1, modify by sqrt alpha
    //========================================================================
    // double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
    //     // Geff/G is the modification factor for the Poisson equation.
    //     // At the linear level (k=0), this should be the sqrt(alpha) factor.
    //     return get_sqrt_alpha(a);
    // }

    //========================================================================
    // Compute the force DPhi from the density field delta (PURE VIRTUAL)
    // The implementation below uses the linear GeffOverG coupling.
    //========================================================================
    // void compute_force(double a,
    //                    double H0Box,
    //                    FFTWGrid<NDIM> & density_fourier,
    //                    std::string density_assignment_method_used,
    //                    std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

    //     const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        
    //     // Define the fifth-force coupling: delta_fifth_force = coupling * delta
    //     auto coupling = [&](double kBox) { 
    //         return GeffOverG(a, kBox / H0Box) - 1.0; 
    //     };
        
    //     FFTWGrid<NDIM> density_fifth_force;

    //     // Compute delta_fifth_force = (Geff/G - 1) * delta
    //     FML::NBODY::compute_delta_fifth_force<NDIM>(density_fourier, density_fifth_force, coupling);
        
    //     // Add on GR component: delta_total = delta + delta_fifth_force
    //     auto Local_nx = density_fourier.get_local_nx();
    //     #ifdef USE_OMP
    //     #pragma omp parallel for
    //     #endif
    //     for (int islice = 0; islice < Local_nx; islice++) {
    //         for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
    //             auto delta = density_fourier.get_fourier_from_index(fourier_index);
    //             auto delta_fifth_force_val = density_fifth_force.get_fourier_from_index(fourier_index);
    //             density_fifth_force.set_fourier_from_index(fourier_index, delta + delta_fifth_force_val);
    //         }
    //     }

    //     // Compute total force from the total density field (stored in density_fifth_force temporarily)
    //     if (this->force_use_finite_difference_force) {
    //       const int stencil_order = this->force_finite_difference_stencil_order;
    //       const int nboundary_cells = stencil_order/2;

    //       FFTWGrid<NDIM> potential_real;
    //       FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(density_fifth_force,
    //           potential_real,
    //           norm_poisson_equation,
    //           nboundary_cells);

    //       FML::NBODY::compute_force_from_potential_real<NDIM>(potential_real,
    //           force_real,
    //           density_assignment_method_used,
    //           stencil_order);

    //     } else {
    //       FML::NBODY::compute_force_from_density_fourier<NDIM>(
    //           density_fifth_force, force_real, density_assignment_method_used, norm_poisson_equation);
    //     }
    // }

    //========================================================================
    // Read parameters
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }

};

#endif
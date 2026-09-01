#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

#include "read_polycrystal.hpp"

// ====================================================
//               Choose Kokkos spaces
// ====================================================
using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename exec_space::memory_space;

// Simulate a crack in a polycrystal
void polycrystalImportExample( const std::string& filename )
{
    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================

    // All grains have the same density
    double density = inputs["density"];

    // Within-grain parameters
    double E_intra = inputs["elastic_modulus"][0];
    double nu_intra = inputs["Poisson's_ratio"][0];
    double G0_intra = inputs["fracture_energy"][0];
    double K_intra = E_intra / ( 3 * ( 1 - 2 * nu_intra ) );

    // Between-grain parameters
    double E_inter = inputs["elastic_modulus"][1];
    double nu_inter = inputs["Poisson's_ratio"][1];
    double G0_inter = inputs["fracture_energy"][1];
    double K_inter = E_inter / ( 3 * ( 1 - 2 * nu_inter ) );

    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                Polycrystal grains
    // ====================================================

    // For now just load grain IDs and domain size;
    // nearest-neighbor interpolation is performed later
    // during initialization
    Kokkos::Array<double, 3> low_corner;
    Kokkos::Array<double, 3> high_corner;
    Kokkos::Array<int, 3> grain_grid_shape;
    std::string grain_file = inputs["grain_file"];
    auto grainIDs = loadGrainIDs<memory_space>( grain_file, low_corner,
                                                high_corner, grain_grid_shape );

    // Calculate loaded grid spacing for interpolation to use later
    Kokkos::Array<double, 3> grain_dx;
    grain_dx[0] = ( high_corner[0] - low_corner[0] ) / grain_grid_shape[0];
    grain_dx[1] = ( high_corner[1] - low_corner[1] ) / grain_grid_shape[1];
    grain_dx[2] = ( high_corner[2] - low_corner[2] ) / grain_grid_shape[2];

    // ====================================================
    //                   Force models
    // ====================================================
    using model_type = CabanaPD::PMB;

    // Grain force models
    CabanaPD::ForceModel force_model_intra( model_type{}, horizon, K_intra,
                                            G0_intra );
    CabanaPD::ForceModel force_model_inter( model_type{}, horizon, K_inter,
                                            G0_inter );

    // ====================================================
    //                 Particle generation
    // ====================================================
    CabanaPD::Particles particles( memory_space{}, model_type{} );

    // Use geometry loaded from grain file
    std::array<double, 3> low_corner_std, high_corner_std;
    low_corner_std[0] = low_corner[0];
    low_corner_std[1] = low_corner[1];
    low_corner_std[2] = low_corner[2];
    high_corner_std[0] = high_corner[0];
    high_corner_std[1] = high_corner[1];
    high_corner_std[2] = high_corner[2];

    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( horizon /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.
    particles.domain( low_corner_std, high_corner_std, num_cells, halo_width );
    particles.create( exec_space{}, Cabana::InitRandom{}, particles, 0, false );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    // The bottom (fixed) plane
    CabanaPD::Region<CabanaPD::RectangularPrism> bottomPlane(
        low_corner[0], high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], low_corner[2] + horizon );

    // The top (moving) plane
    CabanaPD::Region<CabanaPD::RectangularPrism> topPlane(
        low_corner[0], high_corner[0], low_corner[1], high_corner[1],
        high_corner[2] - horizon, high_corner[2] + horizon );

    // The square traction region
    double tractionRegionSize = inputs["traction_region"];
    double midX = ( low_corner[0] + high_corner[0] ) / 2.0;
    double midY = ( low_corner[1] + high_corner[1] ) / 2.0;
    CabanaPD::Region<CabanaPD::RectangularPrism> forcePlane(
        midX - tractionRegionSize / 2.0, midX + tractionRegionSize / 2.0,
        midY - tractionRegionSize / 2.0, midY + tractionRegionSize / 2.0,
        high_corner[2] - horizon, high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();
    auto type = particles.sliceType();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // No-fail zone--add extra no-fail layer outside of boundary regions
        if ( x( pid, 2 ) <= bottomPlane.high[2] + horizon )
            nofail( pid ) = 1;

        // Nearest-neighbor interpolation to get grain type
        int xGrainIndex =
            Kokkos::floor( ( x( pid, 0 ) - low_corner[0] ) / grain_dx[0] );
        xGrainIndex = xGrainIndex < grain_grid_shape[0]
                          ? xGrainIndex
                          : grain_grid_shape[0] - 1;

        int yGrainIndex =
            Kokkos::floor( ( x( pid, 1 ) - low_corner[1] ) / grain_dx[1] );
        yGrainIndex = yGrainIndex < grain_grid_shape[1]
                          ? yGrainIndex
                          : grain_grid_shape[1] - 1;

        int zGrainIndex =
            Kokkos::floor( ( x( pid, 2 ) - low_corner[2] ) / grain_dx[2] );
        zGrainIndex = zGrainIndex < grain_grid_shape[2]
                          ? zGrainIndex
                          : grain_grid_shape[2] - 1;

        // Set density and material type
        type( pid ) = grainIDs( zGrainIndex, xGrainIndex, yGrainIndex );
        rho( pid ) = density;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::BinaryIndexing indexing;
    auto models = CabanaPD::createMultiForceModel(
        particles, indexing, force_model_intra, force_model_inter );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["max_traction"];
    double b0 = sigma0 / horizon;
    double halfCycleInterval = inputs["half_cycle_interval"];
    double db0_dt = b0 / halfCycleInterval;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();

    // Apply periodic triangle traction on upper plane and
    // fixed boundary condition (zero displacement) on lower plane
    auto bc_fn = KOKKOS_LAMBDA( const int pid, const double t )
    {
        if ( x( pid, 2 ) <= bottomPlane.high[2] )
        {
            u( pid, 0 ) = 0.0;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
        else
        {
            int halfCycle = Kokkos::floor( t / halfCycleInterval );
            int isLastHalf = ( halfCycle % 2 );
            double halfCycleStart =
                static_cast<double>( halfCycle ) * halfCycleInterval;
            double slope = db0_dt * static_cast<double>( 1 - 2 * isLastHalf );
            double intercept = b0 * static_cast<double>( isLastHalf );
            f( pid, 2 ) += slope * ( t - halfCycleStart ) + intercept;
        }
    };
    auto bc = createBoundaryCondition( bc_fn, exec_space{}, solver.particles,
                                       true, forcePlane, bottomPlane );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    polycrystalImportExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

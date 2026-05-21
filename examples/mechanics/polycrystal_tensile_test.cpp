#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

constexpr std::size_t NUM_GRAINS = 16;
constexpr double PI = 3.141592653589793238462643383;

// Get flat index into ND array
template <std::size_t n>
int indexND( const std::array<int, n>& index, const std::array<int, n>& shape )
{
    int outIndex = 0;
    int stride = 1;

    for ( int axis = n - 1; axis >= 0; --axis )
    {
        outIndex += index[axis] * stride;
        stride *= shape[axis];
    }

    return outIndex;
}

// Check if ND array multi-index is valid
template <std::size_t n>
bool isValid( const std::array<int, n>& idx, const std::array<int, n>& shape )
{
    for ( int i = 0; i < n; ++i )
    {
        if ( idx[i] < 0 || idx[i] >= shape[i] )
        {
            return false;
        }
    }
    return true;
}

// Recursive helper function for makeNeighborRelativeIndices
template <std::size_t n>
void makeRelativeIndicesRecursive(
    int axis, std::array<int, n>& curIndex,
    std::vector<std::array<int, n>>& outRelativeIndices )
{
    int maxDist = std::ceil( std::sqrt( static_cast<double>( n ) ) );
    for ( int i = -maxDist; i <= maxDist; ++i )
    {
        curIndex[axis] = i;
        if ( axis < n - 1 )
        {
            makeRelativeIndicesRecursive( axis + 1, curIndex,
                                          outRelativeIndices );
        }
        else
        {
            outRelativeIndices.push_back( curIndex );
        }
    }
}

// Get ND array neighbor relative multi-indices
template <std::size_t n>
void makeNeighborRelativeIndices(
    std::vector<std::array<int, n>>& outRelativeIndices )
{
    std::array<int, n> curIndex = {};
    makeRelativeIndicesRecursive( 0, curIndex, outRelativeIndices );
}

template <std::size_t n>
double distSquared( const std::array<double, n>& a,
                    const std::array<double, n>& b )
{
    double r2 = 0.0;
    for ( int axis = 0; axis < n; ++axis )
    {
        r2 += ( a[axis] - b[axis] ) * ( a[axis] - b[axis] );
    }
    return r2;
}

// n-dimensional Poisson disc sampling in a rectangular prism domain
template <std::size_t n, class RNGType>
void poissonDiscSampling( const std::array<double, n>& extent, double r, int k,
                          std::vector<std::array<double, n>>& outPoints,
                          RNGType& gen )
{
    // Precompute reused values for sampling
    double rn = std::pow( r, static_cast<double>( n ) );
    double rn2 = std::pow( 2.0 * r, static_cast<double>( n ) );

    // Calculate shape of grid
    double cellSize = r / std::sqrt( static_cast<double>( n ) );
    std::array<int, n> gridShape = {};
    int totalCells = 1;
    for ( int axis = 0; axis < n; ++axis )
    {
        double axisCells = std::ceil( extent[axis] / cellSize );

        // Add an extra cell in each direction just in case
        gridShape[axis] = 1 + static_cast<int>( axisCells );
        totalCells *= gridShape[axis];
    }

    // Initialize empty grid (as flat array)
    std::vector<int> grid( totalCells, -1 );

    // Calculate grid search relative indices
    std::vector<std::array<int, n>> nbrIndicesRel;
    makeNeighborRelativeIndices( nbrIndicesRel );

    // Choose first point
    std::uniform_real_distribution<double> coordDist( 0.0, 1.0 );
    auto coordGen = std::bind( coordDist, gen );

    std::array<double, n> x0 = {};
    std::array<int, n> idx0 = {};
    for ( int axis = 0; axis < n; ++axis )
    {
        x0[axis] = coordGen() * extent[axis];
        idx0[axis] = static_cast<int>( std::floor( x0[axis] / cellSize ) );
    };

    grid[indexND( idx0, gridShape )] = 0; // Store x0 location in grid
    outPoints.push_back( x0 );

    // Initialize active set
    std::vector<int> activeSet = { 0 };

    while ( activeSet.size() > 0 )
    {
        std::uniform_int_distribution<int> pointDist( 0, activeSet.size() - 1 );
        int seedIndexInActive = pointDist( gen );
        int seedIndex = activeSet[seedIndexInActive];
        std::array<double, n> seedX = outPoints[seedIndex];

        bool addedPoint = false;

        for ( int i = 0; i < k; ++i )
        {
            std::array<double, n> x = {};
            std::array<int, n> idx = {};

            // Use inverse method to sample distance from seed
            double xR = std::pow( rn + coordGen() * ( rn2 - rn ),
                                  1.0 / static_cast<double>( n ) );

            // Sample direction cosine angles uniformly in [0, pi] to get
            // direction
            bool failed = false;
            for ( int axis = 0; axis < n; ++axis )
            {
                x[axis] = seedX[axis] + xR * std::cos( coordGen() * PI );
                idx[axis] =
                    static_cast<int>( std::floor( x[axis] / cellSize ) );
                if ( x[axis] < 0 || x[axis] >= extent[axis] )
                {
                    failed = true;
                    break;
                }
            }
            if ( failed )
            {
                continue;
            }

            // Check if point is too close to any existing points
            std::array<int, n> checkIdx = idx;
            bool isClose = false;

            for ( const std::array<int, n>& relIdx : nbrIndicesRel )
            {
                for ( int axis = 0; axis < n; ++axis )
                {
                    checkIdx[axis] = idx[axis] + relIdx[axis];
                }

                if ( !isValid( checkIdx, gridShape ) )
                {
                    continue;
                }

                int checkPoint = grid[indexND( checkIdx, gridShape )];
                if ( checkPoint == -1 )
                {
                    continue;
                }

                const std::array<double, n>& checkX = outPoints[checkPoint];

                if ( distSquared( checkX, x ) > r * r )
                {
                    continue;
                }

                isClose = true;
                break;
            }

            // New point is not too close to any existing, so add it
            if ( !isClose )
            {
                outPoints.push_back( x );
                activeSet.push_back( outPoints.size() - 1 );
                grid[indexND( idx, gridShape )] = outPoints.size() - 1;
                addedPoint = true;
                break;
            }
        }
        // If no point could be generated farther than r from existing points,
        // remove seed point from active set
        if ( !addedPoint )
        {
            activeSet.erase( activeSet.begin() + seedIndexInActive );
        }
    }
}

// Generates numGrains random polycrystal grain centers using
// Poisson disc sampling
template <std::size_t numGrains>
void getPolycrystalGrains(
    const std::array<double, 3>& extent,
    std::array<std::array<double, 3>, numGrains>& outLocations )
{
    // Initialize RNG (for now with FIXED seed)
    std::minstd_rand baseRng;
    baseRng.seed( 12345 );
    std::seed_seq randomSeed{ baseRng(), baseRng(), baseRng(), baseRng(),
                              baseRng(), baseRng(), baseRng(), baseRng() };
    std::mt19937 gen( randomSeed );

    // Generate random, evenly-spaced candidate grain locations
    double volume = extent[0] * extent[1] * extent[2];
    double radius =
        2.0 * std::pow( 0.75 * volume / PI / static_cast<double>( numGrains ),
                        1.0 / 3.0 );
    std::vector<std::array<double, 3>> testPoints;
    do
    {
        testPoints.clear();
        poissonDiscSampling( extent, radius, 30, testPoints, gen );
        radius *= 0.95;
    } while ( testPoints.size() < numGrains );

    // Randomly choose grains locations from candidates
    std::vector<std::size_t> indices( testPoints.size(), 0 );
    std::iota( indices.begin(), indices.end(), 0 );
    std::shuffle( indices.begin(), indices.end(), gen );
    for ( std::size_t i = 0; i < numGrains; ++i )
    {
        outLocations[i] = testPoints[indices[i]];
    }
}

// Simulate a crack in a polycrystal
void polycrystalExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = { inputs["low_corner"][0],
                                         inputs["low_corner"][1],
                                         inputs["low_corner"][2] };
    std::array<double, 3> high_corner = { inputs["high_corner"][0],
                                          inputs["high_corner"][1],
                                          inputs["high_corner"][2] };

    // ====================================================
    //                Material parameters
    // ====================================================
    Kokkos::Array<double, NUM_GRAINS> grainRho;
    for ( int i = 0; i < NUM_GRAINS; ++i )
    {
        grainRho[i] = inputs["density"][i];
    }

    // Within-grain parameters
    double E_within = inputs["elastic_modulus"][0];
    double nu = 0.25;
    double G0_within = inputs["fracture_energy"][0];
    double K_within = E_within / ( 3 * ( 1 - 2 * nu ) );
    double sigma_y_within = inputs["yield_stress"][0];

    // Between-grain parameters
    double E_between = inputs["elastic_modulus"][1];
    double G0_between = inputs["fracture_energy"][1];
    double K_between = E_between / ( 3 * ( 1 - 2 * nu ) );
    double sigma_y_between = inputs["yield_stress"][1];

    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                Polycrystal grains
    // ====================================================
    std::array<double, 3> extent = inputs["system_size"];
    std::array<std::array<double, 3>, NUM_GRAINS> grainPosStd;
    getPolycrystalGrains( extent, grainPosStd );

    // Shift grains relative to low_corner and copy to Kokkos::Array
    Kokkos::Array<Kokkos::Array<double, 3>, NUM_GRAINS> grainPos;
    for ( int i = 0; i < NUM_GRAINS; ++i )
    {
        grainPos[i] = { grainPosStd[i][0] + low_corner[0],
                        grainPosStd[i][1] + low_corner[1],
                        grainPosStd[i][2] + low_corner[2] };
    }

    // ====================================================
    //                   Force models
    // ====================================================
    using model_type = CabanaPD::PMB;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // Grain force models
    CabanaPD::ForceModel force_model_within( model_type{}, mechanics_type{}, 
                                             memory_space{}, horizon, K_within,
                                             G0_within, sigma_y_within);
    CabanaPD::ForceModel force_model_between( model_type{}, mechanics_type{}, 
                                              memory_space{}, horizon, K_between,
                                              G0_between, sigma_y_between);

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto type = particles.sliceType();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Distance squared from nearest grain location
        double distSq = 0.0;
        int grainIndex = 0;
        for ( int i = 0; i < NUM_GRAINS; ++i )
        {
            const Kokkos::Array<double, 3>& pos = grainPos[i];
            double distX = x( pid, 0 ) - pos[0];
            double distY = x( pid, 1 ) - pos[1];
            double distZ = x( pid, 2 ) - pos[2];
            double check = distX * distX + distY * distY + distZ * distZ;
            if ( i == 0 || check < distSq )
            {
                distSq = check;
                grainIndex = i;
            }
        }

        // Density and material type
        type( pid ) = grainIndex;
        rho( pid ) = grainRho[grainIndex];
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::BinaryIndexing indexing;
    auto models = CabanaPD::createMultiForceModel(
        particles, indexing, force_model_within, force_model_between );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.

    // Grip velocity and width
    double v0 = inputs["grip_velocity"];

    // Create region for both grips.
    CabanaPD::Region<CabanaPD::RectangularPrism> right_grip(
        high_corner[0] - horizon, high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> left_grip(
        low_corner[0], low_corner[0] + horizon, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    // Create BC last to ensure ghost particles are included.
    x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        if ( right_grip.inside( x, pid ) )
        {
            u( pid, 0 ) = v0 * t;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
        else if ( left_grip.inside( x, pid ) )
        {
            u( pid, 0 ) = 0.0;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition( disp_func, exec_space{},
                                                 solver.particles, false,
                                                 left_grip, right_grip );

    // ====================================================
    //                      Outputs
    // ====================================================
    auto dx = solver.particles.dx[0];
    auto dy = solver.particles.dx[1];
    auto dz = solver.particles.dx[2];
    auto f = solver.particles.sliceForce();

    // Generate force outputs for right grip to compute stress.
    // Output force on right grip in x-direction.
    auto force_func_x = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 0 ) * dx * dy * dz;
    };
    auto output_fx = CabanaPD::createOutputTimeSeries(
        "output_force_x.txt", inputs, exec_space{}, solver.particles,
        force_func_x, right_grip );

    // Output force on right grip in y-direction.
    auto force_func_y = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 1 ) * dx * dy * dz;
    };
    auto output_fy = CabanaPD::createOutputTimeSeries(
        "output_force_y.txt", inputs, exec_space{}, solver.particles,
        force_func_y, right_grip );

    // Output force on right grip in z-direction.
    auto force_func_z = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 2 ) * dx * dy * dz;
    };
    auto output_fz = CabanaPD::createOutputTimeSeries(
        "output_force_z.txt", inputs, exec_space{}, solver.particles,
        force_func_z, right_grip );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc, output_fx, output_fy, output_fz );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    polycrystalExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

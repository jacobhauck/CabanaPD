#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

constexpr std::size_t NUM_GRAINS = 1000;
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
                          std::array<int, n>& outGridShape,
                          double& outCellSize,
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

    outGridShape = gridShape;
    outCellSize = cellSize;
}

// Generates numGrains random polycrystal grain centers using
// Poisson disc sampling
void getPolycrystalGrains(
    const std::array<double, 3>& extent,
    std::size_t numGrains,
    std::vector<std::array<double, 3>>& outLocations,
    std::array<int, 3>& outGridShape,
    double& outCellSize
)
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
        std::cout << "Sampling with radius " << radius << std::endl;
        poissonDiscSampling( extent, radius, 30, testPoints, outGridShape, outCellSize, gen );
        radius *= 0.95;
        std::cout << "Found points: " << testPoints.size() << std::endl;
    } while ( testPoints.size() < numGrains );
    std::cout << "Found suitable radius" << std::endl;

    // Randomly choose grains locations from candidates
    std::vector<std::size_t> indices( testPoints.size() );
    std::iota( indices.begin(), indices.end(), 0 );
    std::shuffle( indices.begin(), indices.end(), gen );
    outLocations.resize(numGrains);
    for ( std::size_t i = 0; i < numGrains; ++i )
    {
        outLocations[i] = testPoints[indices[i]];
    }
    std::cout << "Subsampled points" << std::endl;
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
    double grainRho = inputs["density"][0];

    // Within-grain parameters
    double E_within = inputs["elastic_modulus"][0];
    double nu_within = inputs["Poisson's_ratio"][0];
    double G0_within = inputs["fracture_energy"][0];
    double K_within = E_within / ( 3 * ( 1 - 2 * nu_within ) );
    double G_within = E_within / ( 2 * ( 1 + nu_within ) );

    // Between-grain parameters
    double E_between = inputs["elastic_modulus"][1];
    double nu_between = inputs["Poisson's_ratio"][1];
    double G0_between = inputs["fracture_energy"][1];
    double K_between = E_between / ( 3 * ( 1 - 2 * nu_between ) );
    double G_between = E_between / ( 2 * ( 1 + nu_between ) );

    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                Polycrystal grains
    // ====================================================
    std::array<double, 3> extent = inputs["system_size"];
    std::vector<std::array<double, 3>> grainPosStd;
    std::array<int, 3> grainGridShape;
    double grainGridCellSize;

    getPolycrystalGrains( extent, NUM_GRAINS, grainPosStd, grainGridShape, grainGridCellSize );
    std::cout << "Called getPolycrystalGrains" << std::endl;
    std::cout << "Grid cell size = " << grainGridCellSize << std::endl;
    
    // Shift grains relative to low_corner and copy to Kokkos::View
    // and also copy grid to Kokkos::View
    Kokkos::View<double*[3], Kokkos::HostSpace> grainPosHost("Host grain position", NUM_GRAINS);
    Kokkos::View<int***, Kokkos::HostSpace> grainGridHost("Host grain grid", grainGridShape[0], grainGridShape[1], grainGridShape[2]);
    std::cout << "Allocated host memory" << std::endl;
    Kokkos::MDRangePolicy grainGridRange({0, 0, 0}, {grainGridShape[0], grainGridShape[1], grainGridShape[2]});
    Kokkos::parallel_for("Init grain grid", grainGridRange, KOKKOS_LAMBDA(int ix, int iy, int iz){
        grainGridHost(ix, iy, iz) = -1;
    });
    std::cout << "Initialized grain grid" << std::endl;

    for ( int i = 0; i < NUM_GRAINS; ++i )
    {
        grainPosHost(i, 0) = grainPosStd[i][0] + low_corner[0];
        grainPosHost(i, 1) = grainPosStd[i][1] + low_corner[1];
        grainPosHost(i, 2) = grainPosStd[i][2] + low_corner[2];
        
        std::array<int, 3> index = {
            static_cast<int>(std::floor(grainPosStd[i][0] / grainGridCellSize)),
            static_cast<int>(std::floor(grainPosStd[i][1] / grainGridCellSize)),
            static_cast<int>(std::floor(grainPosStd[i][2] / grainGridCellSize)),
        };
        grainGridHost(index[0], index[1], index[2]) = i;
    }
    std::cout << "Wrote to grainPosHost and grainGridHost" << std::endl;
    // Now copy from host memory to target memory_space 
    Kokkos::View<double**, memory_space> grainPos("Grain positions", NUM_GRAINS, 3);
    Kokkos::View<int***, memory_space> grainGrid("Grain grid", grainGridShape[0], grainGridShape[1], grainGridShape[2]);
    std::cout << "Allocated grainPos and grainGrid" << std::endl;
    Kokkos::deep_copy(grainPos, grainPosHost);
    Kokkos::deep_copy(grainGrid, grainGridHost);
    std::cout << "Made grains" << std::endl;

    // ====================================================
    //                   Force models
    // ====================================================
    using model_type = CabanaPD::PMB;

    // Grain force models
    CabanaPD::ForceModel force_model_within( model_type{}, horizon, K_within,
                                             G0_within );
    CabanaPD::ForceModel force_model_between( model_type{}, horizon, K_between,
                                              G0_between );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );
    std::cout << "Created particles" << std::endl;

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dy = particles.dx[1];
    CabanaPD::Region<CabanaPD::RectangularPrism> plane1(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> plane2(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1] + dy,
        low_corner[2], high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();
    auto type = particles.sliceType();
    
    int sizeX = grainGridShape[0];
    int sizeY = grainGridShape[1];
    int sizeZ = grainGridShape[2];
    double minX = low_corner[0];
    double minY = low_corner[1];
    double minZ = low_corner[2];

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low[1] + horizon ||
             x( pid, 1 ) >= plane2.high[1] - horizon )
            nofail( pid ) = 1;

        // Distance squared from nearest grain location
        double distSq = 0.0;
        int grainIndex = 0;
        int gx = static_cast<int>(Kokkos::floor((x(pid, 0) - minX) / grainGridCellSize));
        int gy = static_cast<int>(Kokkos::floor((x(pid, 1) - minY) / grainGridCellSize));
        int gz = static_cast<int>(Kokkos::floor((x(pid, 2) - minZ) / grainGridCellSize));

        // Search for any grain in shells of increasing size
        int shellRadius = 0;
        int guessIndex = -1;
        while (guessIndex == -1)
        {
            // X planes
            for(int sy = Kokkos::max(gy - shellRadius, 0); sy <= Kokkos::min(gy + shellRadius, sizeY - 1); ++sy) 
            {
                for(int sz = Kokkos::max(gz - shellRadius, 0); sz <= Kokkos::min(gz + shellRadius, sizeZ - 1); ++sz) 
                {
                    int sx = Kokkos::max(gx - shellRadius, 0);
                    guessIndex = grainGrid(sx, sy, sz);
                    if (guessIndex > 0)
                    {
                        sy = Kokkos::min(gy + shellRadius, sizeY - 1) + 1;
                        break;
                    }

                    sx = Kokkos::min(gx + shellRadius, sizeX - 1);
                    guessIndex = grainGrid(sx, sy, sz);
                    if (guessIndex > 0)
                    {
                        sy = Kokkos::min(gy + shellRadius, sizeY - 1) + 1;
                        break;
                    }
                }
            }

            if (guessIndex > 0)
                break;

            // Y planes
            for(int sx = Kokkos::max(gx - shellRadius, 0); sx <= Kokkos::min(gx + shellRadius, sizeX - 1); ++sx) 
            {
                for(int sz = Kokkos::max(gz - shellRadius, 0); sz <= Kokkos::min(gz + shellRadius, sizeZ - 1); ++sz) 
                {
                    int sy = Kokkos::max(gy - shellRadius, 0);
                    guessIndex = grainGrid(sx, sy, sz);
                    if (guessIndex > 0)
                    {
                        sx = Kokkos::min(gx + shellRadius, sizeX - 1) + 1;
                        break;
                    }

                    sy = Kokkos::min(gy + shellRadius, sizeY - 1);
                    guessIndex = grainGrid(sx, sy, sz);
                    if (guessIndex > 0)
                    {
                        sx = Kokkos::min(gx + shellRadius, sizeX - 1) + 1;
                        break;
                    }
                }
            }

            if (guessIndex > 0)
                break;

            // Z planes
            for(int sy = Kokkos::max(gy - shellRadius, 0); sy <= Kokkos::min(gy + shellRadius, sizeY - 1); ++sy) 
            {
                for(int sx = Kokkos::max(gx - shellRadius, 0); sx <= Kokkos::min(gx + shellRadius, sizeX - 1); ++sx) 
                {
                    int sz = Kokkos::max(gz - shellRadius, 0);
                    guessIndex = grainGrid(sx, sy, sz);
                    if (guessIndex > 0)
                    {
                        sy = Kokkos::min(gy + shellRadius, sizeY - 1) + 1;
                        break;
                    }

                    sz = Kokkos::min(gz + shellRadius, sizeZ - 1);
                    guessIndex = grainGrid(sx, sy, sz);
                    if (guessIndex > 0)
                    {
                        sy = Kokkos::min(gy + shellRadius, sizeY - 1) + 1;
                        break;
                    }
                }
            }

            ++shellRadius;
        }

        // Now find the nearest grain center
        double guessDX = grainPos(guessIndex, 0) - x(pid, 0);
        double guessDY = grainPos(guessIndex, 1) - x(pid, 1);
        double guessDZ = grainPos(guessIndex, 2) - x(pid, 2);
        double dist = Kokkos::sqrt(guessDX * guessDX + guessDY * guessDY + guessDZ * guessDZ);
        int gridSearchDist = static_cast<int>(Kokkos::ceil(dist / grainGridCellSize));

        double closestDist = dist;
        int closestIndex = guessIndex;
        for(int sx = Kokkos::max(gx - gridSearchDist, 0); sx <= Kokkos::min(gx + gridSearchDist, sizeX - 1); ++sx)
        {
            for(int sy = Kokkos::max(gy - gridSearchDist, 0); sy <= Kokkos::min(gy + gridSearchDist, sizeY - 1); ++sy)
            {
                for(int sz = Kokkos::max(gz - gridSearchDist, 0); sz <= Kokkos::min(gz + gridSearchDist, sizeZ - 1); ++sz)
                {
                    int testIndex = grainGrid(sx, sy, sz);
                    double testDX = grainPos(testIndex, 0) - x(pid, 0);
                    double testDY = grainPos(testIndex, 1) - x(pid, 1);
                    double testDZ = grainPos(testIndex, 2) - x(pid, 2);
                    double testDist = Kokkos::sqrt(testDX * testDX + testDY * testDY + testDZ * testDZ);
                    if (testDist < closestDist)
                    {
                        closestDist = testDist;
                        closestIndex = testIndex;
                    }
                }
            }
        }

        // Set material type and density
        type( pid ) = closestIndex;
        rho( pid ) = grainRho;
    };
    particles.update( exec_space{}, init_functor );
    std::cout << "Set grains" << std::endl;

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
    double v0 = inputs["speed"];
    double maxY = high_corner[1];
    double midY = (minY + maxY) / 2.0;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    // Create symmetric displacement boundary condition
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double ypos = x( pid, 1 ) - midY;
        double sign = 0.0;
        double start = 0.0;
        if( ypos > 0 )
        {
            start = maxY;
            sign = 1.0;
        }
        else
        {
            start = minY;
            sign = -1.0;
        }
        x( pid, 1 ) = start + sign * v0 * t;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, plane1, plane2 );

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

    polycrystalExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}

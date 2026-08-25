#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// ====================================================
//               Choose Kokkos spaces
// ====================================================
using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = typename exec_space::memory_space;

// Given a string s, split into tokens on separator
// Modifies outTokens to hold the separated values
void splitString( const std::string& s, char separator,
                  std::vector<std::string>& outTokens )
{
    // Remove any existing values from output vector
    outTokens.clear();

    // Find tokens and add to output
    std::size_t curStart = 0;
    std::size_t sepIndex;
    while ( ( sepIndex = s.find( separator, curStart ) ) != std::string::npos )
    {
        outTokens.push_back( s.substr( curStart, sepIndex - curStart ) );
        curStart = sepIndex + 1;
    }
    outTokens.push_back( s.substr( curStart ) );
}

// Read and discard "n_lines" lines of data from the file
void skipLines( std::ifstream& input_data_stream, int n_lines )
{
    std::string dummy_str;
    for ( int line = 0; line < n_lines; line++ )
    {
        getline( input_data_stream, dummy_str );
    }
}

// Swaps bits for a variable of type SwapType
template <typename SwapType>
void swapEndian( SwapType& var )
{
    // Cast var into a char array (bit values)
    char* varArray = reinterpret_cast<char*>( &var );

    // Size of char array
    int size = sizeof( var );

    // Swap the "ith" bit with the bit "i" from the end of the array
    for ( long i = 0; i < static_cast<long>( size / 2 ); i++ )
    {
        std::swap( varArray[size - 1 - i], varArray[i] );
    }
}

// Reads binary data of the type ReadType, optionally swapping the endian format
template <typename ReadType>
ReadType readBinaryData( std::ifstream& inStream, bool doSwapEndian = false )
{
    unsigned char temp[sizeof( ReadType )];
    inStream.read( reinterpret_cast<char*>( temp ), sizeof( ReadType ) );
    if ( doSwapEndian )
    {
        swapEndian( temp );
    }
    ReadType read_value = reinterpret_cast<ReadType&>( temp );
    return read_value;
}

// Parse space-separated ASCII data loaded into the string stream
template <typename ReadType>
ReadType parseASCIIData( std::istringstream& ss )
{
    ReadType read_value;
    ss >> read_value;
    return read_value;
}

// Reads portion of a paraview file and places data in the appropriate data
// structure ASCII data at each Z value is separated by a newline
template <typename ReadViewType>
ReadViewType readASCIIField( std::ifstream& inStream, int nx, int ny, int nz,
                             const std::string& label )
{
    ReadViewType field( Kokkos::ViewAllocateWithoutInitializing( label ), nz,
                        nx, ny );
    using value_type = typename ReadViewType::value_type;

    for ( int k = 0; k < nz; k++ )
    {
        // Get line from file
        std::string line;
        getline( inStream, line );

        // Parse string at spaces
        std::istringstream ss( line );
        for ( int j = 0; j < ny; j++ )
        {
            for ( int i = 0; i < nx; i++ )
            {
                field( k, i, j ) = parseASCIIData<value_type>( ss );
            }
        }
    }
    return field;
}

// Reads binary string of type read_datatype from a paraview file, converts
// field to the appropriate type to match read_view_type_3d_host (i.e,
// value_type), and place data in the appropriate data structure Each field
// consists of a single binary string (no newlines) Store converted values in
// view - LayerID data is a short int, GrainID data is an int In some older vtk
// files, LayerID may have been stored as an int and should be converted
template <typename ReadViewType, typename ReadType>
ReadViewType readBinaryField( std::ifstream& input_data_stream, int nx, int ny,
                              int nz, const std::string& label )
{
    ReadViewType field( Kokkos::ViewAllocateWithoutInitializing( label ), nz,
                        nx, ny );
    using value_type = typename ReadViewType::value_type;

    for ( int k = 0; k < nz; k++ )
    {
        for ( int j = 0; j < ny; j++ )
        {
            for ( int i = 0; i < nx; i++ )
            {
                ReadType parsed_value =
                    readBinaryData<ReadType>( input_data_stream, true );
                field( k, i, j ) = static_cast<value_type>( parsed_value );
            }
        }
    }
    return field;
}

// Read the grain structure from a .vtk located at fileName
// Returns lower corner [x, y, z] and upper corner [x, y, z]
// of the grain structure in outLowCorner and outHighCorner
// as well as the grid shape in outGridShape. The grain data
// is returned as the main return value in a Kokkos view.
Kokkos::View<int***, memory_space>
loadGrainIDs( const std::string& fileName,
              Kokkos::Array<double, 3>& outLowCorner,
              Kokkos::Array<double, 3>& outHighCorner,
              Kokkos::Array<int, 3>& outGridShape )
{
    // Open input file
    std::ifstream grainFile;
    grainFile.open( fileName );

    // Ignore first two header lines
    skipLines( grainFile, 2 );

    // Is this data binary or ASCII
    std::string read_line;
    getline( grainFile, read_line );
    const bool isBinary = ( read_line.find( "BINARY" ) != std::string::npos );

    // Ignore line
    skipLines( grainFile, 1 );

    // Get nx, ny and nz
    std::vector<std::string> dims_read;
    getline( grainFile, read_line );
    splitString( read_line, ' ', dims_read );
    const int nx = std::stoi( dims_read[1] );
    const int ny = std::stoi( dims_read[2] );
    const int nz = std::stoi( dims_read[3] );
    outGridShape[0] = nx;
    outGridShape[1] = ny;
    outGridShape[2] = nz;

    // Get origin location
    std::vector<std::string> origin_read;
    getline( grainFile, read_line );
    splitString( read_line, ' ', origin_read );
    outLowCorner[0] = std::stod( origin_read[1] );
    outLowCorner[1] = std::stod( origin_read[2] );
    outLowCorner[2] = std::stod( origin_read[3] );

    // Get voxel spacing
    std::vector<std::string> vox_spacing_read;
    getline( grainFile, read_line );
    splitString( read_line, ' ', vox_spacing_read );
    const double dx = std::stod( vox_spacing_read[1] );
    const double dy = std::stod( vox_spacing_read[2] );
    const double dz = std::stod( vox_spacing_read[3] );

    // Compute domain size
    outHighCorner[0] = outLowCorner[0] + outGridShape[0] * dx;
    outHighCorner[1] = outLowCorner[1] + outGridShape[1] * dy;
    outHighCorner[2] = outLowCorner[2] + outGridShape[2] * dz;

    // Ignore line
    skipLines( grainFile, 1 );

    // Ensure data is of type integer
    getline( grainFile, read_line );
    if ( !read_line.find( "int" ) )
    {
        throw std::runtime_error( "Error: grain ID data must be type int" );
    }

    // Ignore line
    skipLines( grainFile, 1 );

    // Read grain ID data from file into a host view
    Kokkos::View<int***, Kokkos::HostSpace> grainIDsHost(
        Kokkos::ViewAllocateWithoutInitializing( "GrainID Host" ), nz, nx, ny );

    if ( isBinary )
    {
        grainIDsHost =
            readBinaryField<Kokkos::View<int***, Kokkos::HostSpace>, int>(
                grainFile, nx, ny, nz, "GrainID" );
    }
    else
    {
        grainIDsHost = readASCIIField<Kokkos::View<int***, Kokkos::HostSpace>>(
            grainFile, nx, ny, nz, "GrainID" );
    }

    // Copy to memory_space
    Kokkos::View<int***, memory_space> grainIDs( "Grain IDs", nz, nx, ny );
    Kokkos::deep_copy( grainIDs, grainIDsHost );

    Kokkos::MDRangePolicy rangePolicy( { 0, 0, 0 }, { nz, nx, ny } );
    int minimum;
    Kokkos::Min<int> minimumReducer( minimum );
    Kokkos::parallel_reduce(
        "Calculate min index", rangePolicy,
        KOKKOS_LAMBDA( int z, int x, int y, int& lmin ) {
            minimumReducer.join( lmin, grainIDs( z, x, y ) );
        },
        minimumReducer );

    Kokkos::parallel_for(
        "Shift to start indices from 0", rangePolicy,
        KOKKOS_LAMBDA( int z, int x, int y ) {
            grainIDs( z, x, y ) -= minimum;
        } );

    return grainIDs;
}

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

    // For now just load grain IDs and domain size;
    // nearest-neighbor interpolation is performed later
    // during intialization
    Kokkos::Array<double, 3> low_corner;
    Kokkos::Array<double, 3> high_corner;
    Kokkos::Array<int, 3> grain_grid_shape;
    Kokkos::View<int***, memory_space> grainIDs = loadGrainIDs(
        inputs["grain_file"], low_corner, high_corner, grain_grid_shape );

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
    CabanaPD::ForceModel force_model_within( model_type{}, horizon, K_within,
                                             G0_within );
    CabanaPD::ForceModel force_model_between( model_type{}, horizon, K_between,
                                              G0_between );

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
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();
    auto type = particles.sliceType();
    double maxY = high_corner[1];
    double minY = low_corner[1];

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // No-fail zone--add extra no-fail layer outside of boundary regions
        if ( x( pid, 1 ) >= maxY - horizon || x( pid, 1 ) <= minY + horizon )
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
        particles, indexing, force_model_within, force_model_between );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double v0 = inputs["speed"];
    double midY = ( minY + maxY ) / 2.0;
    auto u = solver.particles.sliceDisplacement();
    x = solver.particles.sliceReferencePosition();

    auto bc_op = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double ypos = x( pid, 1 ) - midY;
        double sign = 0.0;
        if ( ypos < 0.0 )
        {
            sign = -1.0;
        }
        else
        {
            sign = 1.0;
        }
        u( pid, 1 ) = sign * v0 * t;
    };
    double dy = particles.dx[1];
    CabanaPD::Region<CabanaPD::RectangularPrism> bcPlaneBottom(
        low_corner[0], high_corner[0], low_corner[1], low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> bcPlaneTop(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1],
        low_corner[2], high_corner[2] );
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, bcPlaneBottom, bcPlaneTop );

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

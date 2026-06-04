#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

//*****************************************************************************/
// Remove whitespace from "line", optional argument to take only portion of the line after position "pos"
std::string removeWhitespace(std::string line, int pos) {

    std::string val = line.substr(pos + 1, std::string::npos);
    std::regex r("\\s+");
    val = std::regex_replace(val, r, "");
    return val;
}

// Check if a string is Y (true) or N (false)
bool getInputBool(std::string val_input) {
    std::string val = removeWhitespace(val_input);
    if (val == "N") {
        return false;
    }
    else if (val == "Y") {
        return true;
    }
    else {
        std::string error = "Input \"" + val + "\" must be \"Y\" or \"N\".";
        throw std::runtime_error(error);
    }
}

// Convert string "val_input" to base 10 integer
int getInputInt(std::string val_input) {
    int IntFromString = stoi(val_input, nullptr, 10);
    return IntFromString;
}

// Convert string "val_input" to float value multiplied by 10^(factor)
float getInputFloat(std::string val_input, int factor) {
    float FloatFromString = atof(val_input.c_str()) * pow(10, factor);
    return FloatFromString;
}

// Convert string "val_input" to double value multiplied by 10^(factor)
double getInputDouble(std::string val_input, int factor) {
    double DoubleFromString = std::stod(val_input.c_str()) * pow(10, factor);
    return DoubleFromString;
}

// Given a string ("line"), parse at "separator" (commas used by default)
// Modifies "parsed_line" to hold the separated values
// expected_num_values may be larger than parsed_line_size, if only a portion of the line is being parsed
void splitString(const std::string line, std::vector<std::string> &parsed_line, std::size_t expected_num_values,
                 char separator) {
    // Make sure the right number of values are present on the line - one more than the number of separators
    std::size_t actual_num_values = std::count(line.begin(), line.end(), separator) + 1;
    if (expected_num_values != actual_num_values) {
        std::string error = "Error: Expected " + std::to_string(expected_num_values) +
                            " values while reading file; but " + std::to_string(actual_num_values) + " were found";
        throw std::runtime_error(error);
    }
    // Separate the line into its components, now that the number of values has been checked
    std::size_t parsed_line_size = parsed_line.size();
    // Make a copy that we can modify
    std::string line_copy = line;
    for (std::size_t n = 0; n < parsed_line_size - 1; n++) {
        std::size_t pos = line_copy.find(separator);
        parsed_line[n] = line_copy.substr(0, pos);
        line_copy = line_copy.substr(pos + 1, std::string::npos);
    }
    parsed_line[parsed_line_size - 1] = line_copy;
}

// Reads a line from an input file stream and outputs the components of the line split at "separator" (spaces used by
// default). By default, expects 4 components of the line to separate (used in parsing vtk header data)
std::vector<std::string> splitString(std::ifstream &input_data_stream, std::size_t expected_num_values,
                                     char separator) {

    std::string line;
    std::vector<std::string> parsed_line(expected_num_values);
    getline(input_data_stream, line);
    splitString(line, parsed_line, expected_num_values, separator);
    return parsed_line;
}

bool checkFileExists(const std::string path, const int id, const bool error) {
    std::ifstream stream;
    stream.open(path);
    if (!(stream.is_open())) {
        stream.close();
        if (error)
            throw std::runtime_error("Could not locate/open \"" + path + "\"");
        else
            return false;
    }
    stream.close();
    if (id == 0)
        std::cout << "Opened \"" << path << "\"" << std::endl;
    return true;
}

std::string checkFileInstalled(const std::string name, const int id) {
    // Path to file. Prefer installed location; if not installed use source location.
    std::string path = ExaCA_DATA_INSTALL;
    std::string file = path + "/" + name;
    bool files_installed = checkFileExists(file, id, false);
    if (!files_installed) {
        // If full file path, just use it.
        if (name.substr(0, 1) == "/") {
            file = name;
        }
        // If a relative path, it has to be with respect to the source path.
        else {
            path = ExaCA_DATA_SOURCE;
            file = path + "/" + name;
        }
        checkFileExists(file, id);
    }
    return file;
}

// Make sure file contains data
void checkFileNotEmpty(std::string testfilename) {
    std::ifstream testfilestream;
    testfilestream.open(testfilename);
    std::string testline;
    std::getline(testfilestream, testline);
    if (testline.empty())
        throw std::runtime_error("First line of file " + testfilename + " appears empty");
    testfilestream.close();
}

// Check if the temperature data is in ASCII or binary format
bool checkTemperatureFileFormat(std::string tempfile_thislayer) {
    bool binary_input_data;
    std::size_t found = tempfile_thislayer.find(".catemp");
    if (found == std::string::npos)
        binary_input_data = false;
    else
        binary_input_data = true;
    return binary_input_data;
}

// Check to make sure that the 6 expected column names appear in the correct order in the header for this temperature
// file Return the number of columns present - ignore any columns after the 6 of interest
std::size_t checkForHeaderValues(std::string header_line) {

    // Header values from file - number of commas plus one is the size of the header
    std::size_t header_size = std::count(header_line.begin(), header_line.end(), ',') + 1;
    std::vector<std::string> header_values(header_size, "");
    splitString(header_line, header_values, header_size);

    std::vector<std::vector<std::string>> expected_values = {
        {"x",  "x(m)"},
        {"y",  "y(m)"},
        {"z",  "z(m)"},
        {"tm", "tm(s)"},
        {"tl", "tl(s)", "ts", "ts(s)"},
        {"r",  "r(k/s)", "cr", "cr(k/s)"}
    };
    std::size_t num_expected_values = expected_values.size();
    if (num_expected_values > header_size)
        throw std::runtime_error("Error: Fewer values than expected found in temperature file header");

    // Case insensitive comparison
    for (std::size_t n = 0; n < num_expected_values; n++) {
        auto val = removeWhitespace(header_values[n]);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        // Check each header column label against the expected value(s) - throw error if no match
        std::size_t options_size = expected_values[n].size();
        for (std::size_t e = 0; e < options_size; e++) {
            auto ev = expected_values[n][e];
            if (val == ev)
                break;
            else if (e == options_size - 1)
                throw std::runtime_error(ev + " not found in temperature file header");
        }
    }
    return header_size;
}

// Read and discard "n_lines" lines of data from the file
void skipLines(std::ifstream &input_data_stream, const int n_lines) {
    std::string dummy_str;
    for (int line = 0; line < n_lines; line++)
        getline(input_data_stream, dummy_str);
}

 // Initializes Grain ID values where the substrate comes from a file
void initBaseplateGrainID(const int id, const Grid &grid, const int baseplate_size_z) {

    // Parse substrate input file
    std::ifstream substrate;
    substrate.open(_inputs.substrate_filename);
    // Ignore first two header lines
    skipLines(substrate, 2);

    // Is this data binary or ASCII
    std::string read_line;
    getline(substrate, read_line);
    const bool binary_s = (read_line.find("BINARY") != std::string::npos);

    // Ignore line
    skipLines(substrate, 1);

    // Get nx_s, ny_s, and nz_s
    std::vector<std::string> dims_read = splitString(substrate);
    const int nx_s = getInputInt(dims_read[1]);
    const int ny_s = getInputInt(dims_read[2]);
    const int nz_s = getInputInt(dims_read[3]);

    // Get origin location
    std::vector<std::string> org_read = splitString(substrate);
    const double x_min_s = getInputDouble(org_read[1]);
    const double y_min_s = getInputDouble(org_read[2]);
    const double z_min_s = getInputDouble(org_read[3]);

    // Get voxel spacing
    std::vector<std::string> vox_spacing_read = splitString(substrate);
    const double deltax_s = getInputDouble(vox_spacing_read[1]);
    if ((deltax_s != getInputDouble(vox_spacing_read[2])) || (deltax_s != getInputDouble(vox_spacing_read[3])))
        throw std::runtime_error("Error: substrate data must have same spacing in all directions");

    // Ensure substrate dimensions can sufficiently cover the solidification domain
    if (id == 0)
        std::cout << "Substrate dimensions from file are " << nx_s << " by " << ny_s << " by " << nz_s
                    << ", voxel spacing is " << deltax_s << std::endl;
    checkSubstrateBound(x_min_s, nx_s, deltax_s, grid.x_min, grid.x_max, "X", grid.deltax);
    checkSubstrateBound(y_min_s, ny_s, deltax_s, grid.y_min, grid.y_max, "Y", grid.deltax);
    checkSubstrateBound(z_min_s, nz_s, deltax_s, grid.z_min, _inputs.baseplate_top_z, "Z", grid.deltax);

    // Ignore line
    skipLines(substrate, 1);

    // Ensure data is of type integer
    getline(substrate, read_line);
    if (!read_line.find("int"))
        throw std::runtime_error("Error: substrate grain ID data must be type int");

    // Ignore line
    skipLines(substrate, 1);

    // Read grain ID data from file into the host view
    Kokkos::View<int ***, Kokkos::HostSpace> grain_id_s_host(Kokkos::ViewAllocateWithoutInitializing("GrainID_S"),
                                                                nz_s, nx_s, ny_s);
    if (binary_s)
        grain_id_s_host =
            readBinaryField<Kokkos::View<int ***, Kokkos::HostSpace>, int>(substrate, nx_s, ny_s, nz_s, "GrainID");
    else
        grain_id_s_host =
            readASCIIField<Kokkos::View<int ***, Kokkos::HostSpace>>(substrate, nx_s, ny_s, nz_s, "GrainID");
    if (id == 0)
        std::cout << "Successfully read substrate GrainID data from the file" << std::endl;
    substrate.close();
    // Copy host view data to device
    auto grain_id_s = Kokkos::create_mirror_view_and_copy(memory_space(), grain_id_s_host);

    // Assign each CA cell the GrainID of the nearest voxel in the substrate, returning the min and max GrainID
    const int baseplate_top_coord_1D = grid.nx * grid.ny_local * baseplate_size_z;
    // Struct to store reduction result
    Kokkos::MinMax<int>::value_type bounds_grain_id_local;
    Kokkos::MinMax<int> bounds_grain_id_reducer(bounds_grain_id_local);
    // Local copy for lambda capture
    auto grain_id_all_layers_local = grain_id_all_layers;
    auto policy = Kokkos::RangePolicy<execution_space>(0, baseplate_top_coord_1D);
    Kokkos::parallel_reduce(
        "BaseplateInit", policy,
        KOKKOS_LAMBDA(const int &index_all_layers, Kokkos::MinMax<int>::value_type &update) {
            // x, y, z associated with this 1D cell location, with respect to the global simulation bounds
            const int coord_x_global = grid.getCoordX(index_all_layers);
            const int coord_y_global = grid.getCoordY(index_all_layers) + grid.y_offset;
            const int coord_z_global = grid.getCoordZ(index_all_layers);
            const double x_location = grid.x_min + coord_x_global * grid.deltax;
            const double y_location = grid.y_min + coord_y_global * grid.deltax;
            const double z_location = grid.z_min + coord_z_global * grid.deltax;
            // What voxel does this correspond to in the substrate?
            const int coord_x_s = Kokkos::round((x_location - x_min_s) / deltax_s);
            const int coord_y_s = Kokkos::round((y_location - y_min_s) / deltax_s);
            const int coord_z_s = Kokkos::round((z_location - z_min_s) / deltax_s);
            grain_id_all_layers_local(index_all_layers) = grain_id_s(coord_z_s, coord_x_s, coord_y_s);
            Kokkos::MinMaxScalar<int> current{grain_id_all_layers_local(index_all_layers),
                                                grain_id_all_layers_local(index_all_layers)};
            bounds_grain_id_reducer.join(update, current);
        },
        bounds_grain_id_reducer);
    Kokkos::fence();
    // Avoid reusing grain IDs that were already in the baseplate grain structure in future powder layers (positive
    // values) or future nucleation events (negative values)
    const int min_grain_id_local = bounds_grain_id_local.min_val;
    const int max_grain_id_local = bounds_grain_id_local.max_val;
    MPI_Allreduce(&min_grain_id_local, &num_prior_nuclei, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_grain_id_local, &next_layer_first_epitaxial_grain_id, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // First grain ID used in future layers should be 1 beyond the largest used in the substrate, increment by 1
    next_layer_first_epitaxial_grain_id++;
    // The number of negative grain IDs to be reserved is equivalent to the absolute value of the smallest substrate
    // grain ID (this will then be passed to the nucleation constructor to avoid assigning these IDs to new
    // nucleation events)
    num_prior_nuclei = Kokkos::abs(num_prior_nuclei);
    if (id == 0)
        std::cout << "Substrate file read complete: grain id values in baseplate: " << -num_prior_nuclei
                    << " through " << next_layer_first_epitaxial_grain_id - 1 << std::endl;
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

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low[1] + horizon ||
             x( pid, 1 ) >= plane2.high[1] - horizon )
            nofail( pid ) = 1;

        // Distance squared from nearest grain location
        double distSq = 0.0;
        int grainIndex = 0;
        for ( int i = 0; i < NUM_GRAINS; ++i )
        {
            const Kokkos::Array<double, 3>& pos = grainPos[i];
            double dx = x( pid, 0 ) - pos[0];
            double dy = x( pid, 1 ) - pos[1];
            double dz = x( pid, 2 ) - pos[2];
            double check = dx * dx + dy * dy + dz * dz;
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
    double sigma0 = inputs["traction"];
    double b0 = sigma0 / dy;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    // Create a symmetric force BC in the y-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto ypos = x( pid, 1 );
        auto sign = std::abs( ypos ) / ypos;
        f( pid, 1 ) += b0 * sign;
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

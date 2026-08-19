#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "daphne.h"
#include "H5Cpp.h"
#include <getopt.h>
#include <filesystem>

const std::string DATASET_NAME_PREFIX = "data_ch";  // 每个通道数据集的前缀

// Function to parse command-line arguments
void parseArguments(int argc, char* argv[], int &run, std::string &trigger, int &endpoint, int &nwvfms, int &afe, std::vector<int> &channels) {
    std::string errorMessage = " --run <run number, req.> --ep <endpoint, opt.> --trig <trigger type, req.>\n"
                               "--nwvfms <number of waveforms, opt.> --afe <AFE number> --ch <channels (comma-separated)>";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << errorMessage << std::endl;
        exit(1);
    }

    static struct option long_options[] = {
        {"run", required_argument, 0, 'r'},
        {"trig", required_argument, 0, 't'},
        {"ep", required_argument, 0, 'e'},
        {"nwvfms", required_argument, 0, 'n'},
        {"afe", required_argument, 0, 'a'},
        {"ch", required_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "r:t:e:n:a:c:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'r':
                run = std::stoi(optarg);
                break;
            case 't':
                trigger = optarg;
                break;
            case 'e':
                endpoint = std::stoi(optarg);
                break;
            case 'n':
                nwvfms = std::stoi(optarg);
                break;
            case 'a':
                afe = std::stoi(optarg);
                break;
            case 'c': {
                std::string ch_str(optarg);
                size_t pos = 0;
                while ((pos = ch_str.find(',')) != std::string::npos) {
                    channels.push_back(std::stoi(ch_str.substr(0, pos)));
                    ch_str.erase(0, pos + 1);
                }
                channels.push_back(std::stoi(ch_str));
                break;
            }
            case 'h':
                std::cout << "Usage: " << argv[0] << errorMessage;
                exit(0);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << errorMessage;
                exit(1);
        }
    }

    if (trigger != "ext" && trigger != "soft") {
        std::cerr << "Error: The --trig option must be either 'ext' or 'soft'.\n";
        exit(1);
    }
}

// Function to collect data from a single channel
void collectData(int run, int endpoint, int nwvfms, int afe, int ch, bool use_software_trigger, H5::H5File &file) {
    std::string ipaddr = "10.73.137." + std::to_string(endpoint);
    std::string dataset_name = DATASET_NAME_PREFIX + std::to_string(ch);

    Daphne daphne(ipaddr);
    std::vector<int> combined_result;
    hsize_t offset[2] = {0, 0};

    hsize_t initial_dims[2] = {0, 0};
    hsize_t max_dims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    H5::DataSpace dataspace(2, initial_dims, max_dims);
    H5::DataType uint32_type = H5::PredType::NATIVE_UINT;

    H5::DSetCreatPropList propList;
    hsize_t array_length = 3900;
    hsize_t chunk_dims[2] = {1, array_length};
    propList.setChunk(2, chunk_dims);
    propList.setDeflate(2);

    H5::DataSet dataset = file.createDataSet(dataset_name, uint32_type, dataspace, propList);

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::minutes(10);

    int chunk_length = 150;
    unsigned int base_register = 0x40000000;
    unsigned int AFE_hex_base = 0x100000;
    unsigned int Channel_hex_base = 0x10000;
    int length = use_software_trigger ? 4000 : 2000;

    hsize_t Length = static_cast<hsize_t>(length);
    const hsize_t hsizeArray[2] = {Length, static_cast<hsize_t>(nwvfms)};

    unsigned int chunks = length / chunk_length;
    bool use_iterations_limit = true;
    int iterations_limit = nwvfms;
    if (use_iterations_limit) {
        end_time = end_time + std::chrono::minutes(10000);
    }

    int iteration = 0;
    int base_address = base_register + (AFE_hex_base * afe) + (Channel_hex_base * ch);
    unsigned int last_timestamp = 0;
    auto last_time = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() < end_time) {
        if (use_software_trigger) {
            daphne.write_reg(0x2000, {1234});
        }

        unsigned int current_timestamp = daphne.read_reg(0x40500000, 1)[0];

        if (last_timestamp != current_timestamp) {
            combined_result.clear();
            for (unsigned int i = 0; i < chunks; ++i) {
                unsigned int address = base_address + i * chunk_length;
                std::vector<int> doutrec = daphne.read_reg(address, chunk_length);
                combined_result.insert(combined_result.end(), doutrec.begin(), doutrec.end());
            }

            unsigned int new_timestamp = daphne.read_reg(0x40500000, 1)[0];
            if (new_timestamp == current_timestamp) {
                hsize_t current_dims[2] = {offset[0] + 1, offset[1] + array_length};
                dataset.extend(current_dims);

                H5::DataSpace new_dataspace = dataset.getSpace();
                hsize_t batch_size[2] = {1, array_length};
                new_dataspace.selectHyperslab(H5S_SELECT_SET, batch_size, offset);

                hsize_t mem_dims[2] = {1, array_length};
                H5::DataSpace memspace(2, mem_dims);

                dataset.write(combined_result.data(), uint32_type, memspace, new_dataspace);
                offset[0] += 1;

                ++iteration;
                last_timestamp = new_timestamp;
            }
        }

        if (use_iterations_limit && iteration >= iterations_limit) {
            break;
        }
    }

    dataset.close();
    daphne.close_conn();
}

int main(int argc, char* argv[]) {
    int run = 0;
    int endpoint = 110;
    int nwvfms = 500;
    int afe = 0;
    std::vector<int> channels;
    std::string trigger;

    parseArguments(argc, argv, run, trigger, endpoint, nwvfms, afe, channels);
    bool use_software_trigger = (trigger == "soft");

    std::string filename = "data/run" + std::to_string(run) + ".hdf5";
    if (std::filesystem::exists(filename)) {
        std::cerr << "File " << filename << " already exists. Exiting...\n";
        return 1;
    }

    H5::H5File file(filename, H5F_ACC_TRUNC);
    std::vector<std::thread> threads;

    for (int ch : channels) {
        threads.emplace_back(collectData, run, endpoint, nwvfms, afe, ch, use_software_trigger, std::ref(file));
    }

    for (auto& t : threads) {
        t.join();
    }

    file.close();
    return 0;
}

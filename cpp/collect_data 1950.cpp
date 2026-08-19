#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <getopt.h>
#include "daphne.h"
#include "H5Cpp.h"

void parseArguments(int argc, char* argv[], int &run, std::string &trigger, int &endpoint, int &nwvfms, int &afe, std::vector<int> &channels) {
    std::string errorMessage = "\nUsage: ./collect_data --run <num> --trig <ext|soft> --ep <endpoint> --nwvfms <number> --afe <afe> --ch <comma-separated channels>";

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

    int opt;
    while ((opt = getopt_long(argc, argv, "r:t:e:n:a:c:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'r': run = std::stoi(optarg); break;
            case 't': trigger = optarg; break;
            case 'e': endpoint = std::stoi(optarg); break;
            case 'n': nwvfms = std::stoi(optarg); break;
            case 'a': afe = std::stoi(optarg); break;
            case 'c': {
                std::string ch_arg = optarg;
                std::stringstream ss(ch_arg);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    channels.push_back(std::stoi(token));
                }
                break;
            }
            case 'h':
            default:
                std::cerr << errorMessage << std::endl;
                exit(1);
        }
    }

    if (run == -1 || trigger.empty() || channels.empty()) {
        std::cerr << "Missing required arguments." << errorMessage << std::endl;
        exit(1);
    }
}

void acquire_channel(int run, const std::string &trigger, int endpoint, int nwvfms, int afe, int ch) {
    std::string ipaddr = "10.73.137." + std::to_string(endpoint);
    std::string filename = "data/run" + std::to_string(run) + "_ch" + std::to_string(ch) + ".hdf5";

    std::cout << "[CH" << ch << "] Saving to " << filename << std::endl;
    if (std::filesystem::exists(filename)) {
        std::cerr << "File " << filename << " already exists. Skipping...\n";
        return;
    }

    Daphne daphne(ipaddr);
    H5::H5File file(filename, H5F_ACC_TRUNC);
    hsize_t initial_dims[2] = {0, 0};
    hsize_t max_dims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
    H5::DataSpace dataspace(2, initial_dims, max_dims);
    H5::DataType uint32_type = H5::PredType::NATIVE_UINT;

    hsize_t array_length = 3900;
    hsize_t chunk_dims[2] = {1, array_length};
    H5::DSetCreatPropList propList;
    propList.setChunk(2, chunk_dims);
    propList.setDeflate(2);
    H5::DataSet dataset = file.createDataSet("data", uint32_type, dataspace, propList);

    bool use_soft_trig = (trigger == "soft");
    int length = use_soft_trig ? 4000 : 2000;
    int chunk_len = 150;
    int chunks = length / chunk_len;
    unsigned int base_reg = 0x40000000 + 0x100000 * afe + 0x10000 * ch;

    std::vector<int> waveform;
    hsize_t offset[2] = {0, 0};
    unsigned int last_ts = 0;

    int iteration = 0;
    while (iteration < nwvfms) {
        if (use_soft_trig) daphne.write_reg(0x2000, {1234});
        unsigned int ts = daphne.read_reg(0x40500000, 1)[0];
        if (ts == last_ts) continue;

        waveform.clear();
        for (int i = 0; i < chunks; ++i) {
            unsigned int addr = base_reg + i * chunk_len;
            auto data = daphne.read_reg(addr, chunk_len);
            waveform.insert(waveform.end(), data.begin(), data.end());
        }

        unsigned int ts2 = daphne.read_reg(0x40500000, 1)[0];
        if (ts2 != ts) continue;

        hsize_t cur_dims[2] = {offset[0] + 1, offset[1] + array_length};
        dataset.extend(cur_dims);

        H5::DataSpace fspace = dataset.getSpace();
        hsize_t count[2] = {1, array_length};
        fspace.selectHyperslab(H5S_SELECT_SET, count, offset);

        hsize_t mem_dims[2] = {1, array_length};
        H5::DataSpace mspace(2, mem_dims);
        dataset.write(waveform.data(), uint32_type, mspace, fspace);

        offset[0]++;
        last_ts = ts;
        iteration++;

        if (iteration % 50 == 0) {
            std::cout << "[CH" << ch << "] " << iteration << "/" << nwvfms << " waveforms done." << std::endl;
        }
    }

    file.close();
    daphne.close_conn();
    std::cout << "[CH" << ch << "] Finished acquisition." << std::endl;
}

int main(int argc, char* argv[]) {
    int run = -1, endpoint = 110, nwvfms = 500, afe = 0;
    std::string trigger;
    std::vector<int> channels;

    parseArguments(argc, argv, run, trigger, endpoint, nwvfms, afe, channels);

    for (int ch : channels) {
        acquire_channel(run, trigger, endpoint, nwvfms, afe, ch);
    }
    return 0;
}

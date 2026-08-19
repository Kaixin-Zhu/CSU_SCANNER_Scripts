#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <getopt.h>
#include <iomanip>

#include "daphne.h"
#include "H5Cpp.h"

namespace fs = std::filesystem;

// 获取当前系统的实时时间字符串 (格式: YYYYMMDD_HHMMSS)
std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

void parseArguments(int argc, char* argv[], int &run, std::string &trigger, int &endpoint, int &nwvfms, std::vector<int> &afe_list, std::vector<int> &channels) {
    std::string errorMessage = "\nUsage: ./collect_data --run <num> --trig <ext|soft> --ep <endpoint> --nwvfms <number> --afe <comma-separated afes> --ch <comma-separated channels>";

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

    bool user_provided_ch = false;

    int opt;
    while ((opt = getopt_long(argc, argv, "r:t:e:n:a:c:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'r': run = std::stoi(optarg); break;
            case 't': trigger = optarg; break;
            case 'e': endpoint = std::stoi(optarg); break;
            case 'n': nwvfms = std::stoi(optarg); break;
            case 'a': {
                afe_list.clear();
                std::string afe_arg = optarg;
                std::stringstream ss(afe_arg);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    afe_list.push_back(std::stoi(token));
                }
                break;
            }
            case 'c': {
                if (!user_provided_ch) {
                    channels.clear();
                    user_provided_ch = true;
                }
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

    if (run == -1 || trigger.empty() || channels.empty() || afe_list.empty()) {
        std::cerr << "\n[Error] Missing required arguments!" << std::endl;
        std::cerr << "-> Make sure to provide at least: --run and --afe" << errorMessage << std::endl;
        exit(1);
    }
}

// 采集单个通道数据并存入 HDF5 的函数
void acquire_channel(int run, const std::string &trigger, int endpoint, int nwvfms, int afe, int ch, const std::string &time_str) {
    std::string ipaddr = "10.73.137." + std::to_string(endpoint);
    
    // 💡 【核心修改】：构建用来检测重复的前缀，例如 "run1724_afe4_ch7_"
    std::string folder = "data";
    std::string prefix = "run" + std::to_string(run) + "_afe" + std::to_string(afe) + "_ch" + std::to_string(ch) + "_";
    
    // 检查 data 目录下是否已有相同配置的文件（忽略后面的时间戳）
    if (fs::exists(folder) && fs::is_directory(folder)) {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                std::string filename_only = entry.path().filename().string();
                // 如果已存在的文件名是以我们的配置前缀开头的，说明已经采集过
                if (filename_only.rfind(prefix, 0) == 0) {
                    std::cerr << "❌ [Skipping] " << prefix << "* already exists as "
                              << entry.path().string() << ". Repeated collection is BLOCKED!\n";
                    return;
                }
            }
        }
    }

    // 通过检测后，拼接上本次的实时时间戳创建新文件
    std::string filename = folder + "/" + prefix + time_str + ".hdf5";

    std::cout << "[AFE" << afe << " CH" << ch << "] Saving to " << filename << std::endl;

    Daphne daphne(ipaddr);
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    // 在 HDF5 文件内部写入时间戳 Metadata
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace attr_space(H5S_SCALAR);
    H5::Attribute time_attr = file.createAttribute("timestamp", str_type, attr_space);
    time_attr.write(str_type, time_str);

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
    int length = 4000;
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
            std::cout << "[AFE" << afe << " CH" << ch << "] " << iteration << "/" << nwvfms << " waveforms done." << std::endl;
        }
    }

    file.close();
    daphne.close_conn();
    std::cout << "[AFE" << afe << " CH" << ch << "] Finished acquisition." << std::endl;
}

int main(int argc, char* argv[]) {
    int run = -1, endpoint = 110, nwvfms = 10000;
    std::string trigger = "ext";
    std::vector<int> channels = {0, 2, 5, 7};
    std::vector<int> afe_list;

    parseArguments(argc, argv, run, trigger, endpoint, nwvfms, afe_list, channels);

    // 在开始采集循环前捕捉当前的精确时间戳
    std::string current_time = getCurrentTimeString();

    for (int afe : afe_list) {
        std::cout << "\n================ Starting AFE " << afe << " ================" << std::endl;
        for (int ch : channels) {
            acquire_channel(run, trigger, endpoint, nwvfms, afe, ch, current_time);
        }
    }
    
    std::cout << "\nAll requested acquisitions finished!" << std::endl;
    return 0;
}

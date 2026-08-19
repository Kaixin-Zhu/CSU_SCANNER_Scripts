#include <iostream>
#include <vector>
#include <numeric>
#include <string>

// ROOT 头文件
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TApplication.h"

// HighFive 头文件（用于极其简便地在C++中读写HDF5）
// 如果系统没安装，可以使用 sudo apt install libhighfive-dev
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>

// 基线计算区间
const int BASELINE_START = 100;
const int BASELINE_END = 200;

int main(int argc, char** argv) {
    // 启动 ROOT 应用环境，以便能弹出窗口
    TApplication theApp("App", &argc, argv);

    // 1. 指定 HDF5 文件路径（请根据实际情况修改）
    std::string filename = "data/run1726_afe0_ch0.hdf5";
    
    std::cout << "正在读取 HDF5 文件: " << filename << std::endl;

    // 2. 使用 HighFive 读取 HDF5 数据
    std::vector<std::vector<double>> raw_data;
    try {
        HighFive::File file(filename, HighFive::File::ReadOnly);
        HighFive::DataSet dataset = file.getDataSet("data");
        
        // 将 HDF5 中的二维 dataset 读入到嵌套 vector 中
        dataset.read(raw_data);
    } catch (const HighFive::Exception& err) {
        std::cerr << "读取 HDF5 文件失败: " << err.what() << std::endl;
        return 1;
    }

    if (raw_data.empty()) {
        std::cerr << "数据为空！" << std::endl;
        return 1;
    }

    size_t num_wvfms = raw_data.size();
    size_t num_samples = raw_data[0].size();
    std::cout << "成功载入波形数量: " << num_wvfms << ", 每个波形采样点数: " << num_samples << std::endl;

    // 3. 创建 ROOT TH2D 直方图
    // 参数: 名字, 标题, X轴bins, X轴小值, X轴大值, Y轴bins, Y轴小值, Y轴大值
    TH2D* h2 = new TH2D("h2", "Scanner Waveform 2D Histogram (ROOT);Sample Index;Amplitude",
                        num_samples, 0, num_samples,
                        200, -8000, 8000);

    // 4. 处理数据并填充直方图 (Baseline Calibration & Inversion)
    for (size_t i = 0; i < num_wvfms; ++i) {
        // 计算当前波形的基线均值
        double baseline_sum = 0.0;
        int baseline_counts = BASELINE_END - BASELINE_START;
        for (int t = BASELINE_START; t < BASELINE_END; ++t) {
            baseline_sum += raw_data[i][t];
        }
        double baseline = baseline_sum / baseline_counts;

        // 翻转并校准基线，随后填入 TH2D
        for (size_t t = 0; t < num_samples; ++t) {
            // 信号翻转: - (raw - baseline) = baseline - raw
            double corrected_val = baseline - raw_data[i][t];
            
            // 填充到 2D 直方图：X轴为采样点，Y轴为幅值
            h2->Fill(t, corrected_val);
        }
    }

    // 5. 设置 ROOT 样式并绘图
    gStyle->SetOptStat(0);       // 关闭统计信息框
    gStyle->SetPalette(kBird);   // 设置现代彩虹色调（也可以用 kRainbow 或 kJet）

    TCanvas* c1 = new TCanvas("c1", "Waveform Persistence", 1000, 700);
    c1->SetLogz(1);              // ⭐ 关键：Z轴开启对数刻度（类似 Python 的 LogNorm）

    // "colz" 表示绘制 2D 颜色图，并且在右侧自动附带 Colorbar 颜色条
    h2->Draw("colz");

    std::cout << "绘制完成，正在显示窗口..." << std::endl;
    
    // 进入 ROOT 事件循环，保持窗口不关闭
    theApp.Run();

    return 0;
}

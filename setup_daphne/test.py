import ROOT
import numpy as np
from array import array
import time

# 模拟数据流的类
class DataStreamSimulator:
    """
    模拟数据流类，用于生成动态数据。
    实际数据流处理可以替换这里的 `generate_data` 方法。
    """
    def __init__(self, length):
        self.length = length
        self.x_data = np.linspace(0, 10, length)  # x 轴数据
        self.y_data = np.zeros(length)  # 初始化 y 轴数据
        self.timestamp = 0  # 模拟时间戳

    def generate_data(self):
        """
        模拟生成数据流，可以替换为实际数据读取逻辑。
        """
        self.timestamp += 1
        self.y_data = np.sin(self.x_data + self.timestamp * 0.1) + np.random.normal(0, 0.1, self.length)
        return self.x_data, self.y_data, self.timestamp

# 更新画布函数
def update_canvas(canvas, x_data, y_data):
    """
    更新 ROOT 画布并绘制新的图形
    """
    canvas.Clear()  # 清除画布
    graph = ROOT.TGraph(len(x_data), array('d', x_data), array('d', y_data))  # 创建 TGraph
    graph.SetLineColor(ROOT.kBlue)  # 设置线条颜色
    graph.SetMarkerStyle(20)       # 设置点样式
    graph.SetTitle("Dynamic Data Stream;X-axis;Y-axis")  # 设置标题
    graph.Draw("ALP")  # 绘制连线和点
    canvas.Modified()  # 标记画布已修改
    canvas.Update()    # 更新画布

# 创建数据流对象
length = 100  # 数据长度
data_stream = DataStreamSimulator(length)

# 创建 ROOT 画布
canvas = ROOT.TCanvas("canvas", "Real-Time Data Stream Canvas", 800, 600)

# 模拟实时数据处理的主循环
try:
    while True:
        # 获取模拟数据（可替换为实际数据流）
        x_data, y_data, timestamp = data_stream.generate_data()

        # 打印调试信息（可选）
        print(f"Timestamp: {timestamp}, First 5 Y-Data Points: {y_data[:5]}")

        # 更新画布
        update_canvas(canvas, x_data, y_data)

        # 处理 ROOT 的事件循环，确保窗口响应
        ROOT.gSystem.ProcessEvents()

        # 模拟数据刷新间隔
        time.sleep(0.1)

except KeyboardInterrupt:
    print("程序已终止。")

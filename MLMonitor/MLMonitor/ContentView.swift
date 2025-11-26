//
//  ContentView.swift
//  MLMonitor
//
//  Created by Ege Kaya on 26.11.2025.
//

import SwiftUI
import CoreML
internal import Combine

struct ContentView: View {
    @State private var statusMessage = "Initializing..."
    @State private var anomalyScore = "0.00"
    @State private var isMonitoring = false
    @State private var currentCPU = "0%"
    @State private var currentRAM = "0%"
    @State private var scoreColor: Color = .green
    
    // The "Brain" and "Eyes"
    let collector = SystemCollector()
    let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(spacing: 25) {
            Text("🔮 ML System Monitor")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // --- Dashboard ---
            HStack(spacing: 40) {
                MetricView(label: "CPU Load", value: currentCPU)
                MetricView(label: "RAM Usage", value: currentRAM)
            }
            
            Divider().padding(.vertical)
            
            // --- The Verdict ---
            Text("Anomaly Score")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text(anomalyScore)
                .font(.system(size: 50, weight: .heavy, design: .monospaced))
                .foregroundColor(scoreColor)
                .contentTransition(.numericText(value: Double(anomalyScore) ?? 0))
                .animation(.snappy, value: anomalyScore)
            
            Text(statusMessage)
                .font(.caption)
                .foregroundColor(.gray)
            
            // --- Control ---
            Button(action: { isMonitoring.toggle() }) {
                Text(isMonitoring ? "Stop Monitoring" : "Start Live Monitor")
                    .fontWeight(.semibold)
                    .padding()
                    .frame(width: 200)
            }
            .controlSize(.large)
            .tint(isMonitoring ? .red : .blue)
        }
        .padding(40)
        .frame(minWidth: 500, minHeight: 400)
        .onReceive(timer) { _ in
            if isMonitoring {
                runInference()
            }
        }
    }
    
    func runInference() {
        do {
            // 1. Get Real Data
            let metrics = collector.getSystemMetrics() // [CPU, RAM, 0, 0, 0, 0]
            
            // Update UI for raw metrics
            currentCPU = String(format: "%.1f%%", metrics[0] * 100)
            currentRAM = String(format: "%.1f%%", metrics[1] * 100)
            
            // 2. Prepare Input for Model
            let config = MLModelConfiguration()
            let model = try SystemMonitor(configuration: config)
            let inputArray = try MLMultiArray(shape: [1, 6], dataType: .float32)
            
            // Feed data (Standardize manually if needed, but for POC raw is okay)
            for (index, value) in metrics.enumerated() {
                inputArray[index] = NSNumber(value: value)
            }
            
            // 3. Predict
            let input = SystemMonitorInput(input_features: inputArray)
            let output = try model.prediction(input: input)
            
            // 4. Calculate Loss (Anomaly Score)
            var mse: Float = 0.0
            for i in 0..<6 {
                let diff = metrics[i] - output.reconstruction[i].floatValue
                mse += diff * diff
            }
            
            // 5. Update UI
            self.anomalyScore = String(format: "%.4f", mse)
            self.scoreColor = mse > 0.5 ? .red : (mse > 0.2 ? .orange : .green)
            self.statusMessage = "Processing via Apple Neural Engine"
            
        } catch {
            self.statusMessage = "Error: \(error.localizedDescription)"
        }
    }
}

// Helper View for the Dashboard
struct MetricView: View {
    let label: String
    let value: String
    var body: some View {
        VStack {
            Text(label).font(.caption).foregroundColor(.secondary)
            Text(value).font(.title2).fontWeight(.bold)
        }
    }
}

#Preview {
    ContentView()
}

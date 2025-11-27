//
//  ContentView.swift
//  MLMonitor
//
//  Created by Ege Kaya on 26.11.2025.
//

import SwiftUI
import Charts
import Combine

// --- DATA MODELS ---
struct MetricPoint: Identifiable {
    let id = UUID()
    let time: Date
    let score: Double
}

struct LogEvent: Identifiable {
    let id = UUID()
    let time: String
    let message: String
    let type: EventType
    
    enum EventType { case info, warning, danger }
}

// --- MAIN VIEW ---
struct ContentView: View {
    // The Brain (ML)
    @StateObject private var brain = InferenceEngine()
    // The Warden (Battery)
    @StateObject private var power = PowerManager.shared
    
    private let eyes = SystemCollector()
    
    // UI State
    @State private var history: [MetricPoint] = []
    @State private var logs: [LogEvent] = []
    @State private var isMonitoring = false
    @State private var rawMetrics: [Float] = Array(repeating: 0.0, count: 9)
    
    // Timer (Must be @State var so we can update it dynamically)
    @State private var timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    
    var body: some View {
        NavigationSplitView {
            SidebarView(status: brain.modelStatus, monitoring: $isMonitoring)
        } detail: {
            VStack(spacing: 0) {
                // HEADER
                HeaderView(score: brain.currentScore, threshold: brain.alertThreshold, isAnomalous: brain.isAnomalous)
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                
                Divider()
                
                ScrollView {
                    VStack(spacing: 20) {
                        
                        // --- CHART AREA START ---
                        VStack(alignment: .leading) {
                            HStack {
                                Text("Real-time Anomaly Score")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                Spacer()
                                // Show Battery Status Icon
                                if power.isOnBattery {
                                    Label("Eco Mode", systemImage: "leaf.fill")
                                        .font(.caption)
                                        .foregroundColor(.green)
                                }
                            }
                            
                            // 1. The Chart
                            Chart(history) {
                                LineMark(
                                    x: .value("Time", $0.time),
                                    y: .value("MSE", $0.score)
                                )
                                .foregroundStyle(brain.isAnomalous ? .red : .blue)
                                .interpolationMethod(.catmullRom)
                                
                                RuleMark(y: .value("Threshold", brain.alertThreshold))
                                    .foregroundStyle(.orange.opacity(0.5))
                                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [5]))
                            }
                            // 2. FIXED: Modifiers attached DIRECTLY to Chart
                            .chartYScale(domain: 0...max(1.0, brain.currentScore * 1.5))
                            .frame(height: 200)
                            
                        } // End of Chart VStack
                        .padding()
                        .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .controlBackgroundColor)))
                        .shadow(radius: 1)
                        .padding(.horizontal)
                        .padding(.top)
                        // --- CHART AREA END ---

                        
                        // --- DETECTIVE UI START ---
                        if brain.isAnomalous && !brain.topSuspects.isEmpty {
                            VStack(alignment: .leading) {
                                Label("⚠️ Potential Causes", systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.red)
                                    .padding(.bottom, 5)
                                
                                ForEach(brain.topSuspects) { process in
                                    HStack {
                                        Text(process.name)
                                            .font(.caption)
                                            .bold()
                                        Spacer()
                                        Text(String(format: "%.1f%% CPU", process.cpu))
                                            .font(.caption)
                                            .monospacedDigit()
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(4)
                                    .background(Color.red.opacity(0.1))
                                    .cornerRadius(4)
                                }
                            }
                            .padding()
                            .background(RoundedRectangle(cornerRadius: 8).stroke(Color.red, lineWidth: 1))
                            .padding(.horizontal)
                        }
                        // --- DETECTIVE UI END ---
                        
                        
                        // METRICS GRID
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 140))], spacing: 15) {
                            MetricCard(title: "CPU Load", value: String(format: "%.1f%%", rawMetrics[0]), icon: "cpu")
                            MetricCard(title: "Memory", value: String(format: "%.1f%%", rawMetrics[1]), icon: "memorychip")
                            MetricCard(title: "System Load", value: String(format: "%.2f", rawMetrics[2]), icon: "gauge")
                            MetricCard(title: "Network Up", value: formatBytes(rawMetrics[3]), icon: "arrow.up.circle")
                            MetricCard(title: "Network Down", value: formatBytes(rawMetrics[4]), icon: "arrow.down.circle")
                            if rawMetrics.count > 8 {
                                MetricCard(title: "Threads", value: String(format: "%.0f", rawMetrics[8]), icon: "text.alignleft")
                            }
                        }
                        .padding(.horizontal)
                        
                        // LOGS
                        VStack(alignment: .leading) {
                            Text("Recent Events")
                                .font(.headline)
                                .foregroundColor(.secondary)
                            
                            ForEach(logs.prefix(5)) { log in
                                HStack {
                                    Circle()
                                        .fill(colorForEvent(log.type))
                                        .frame(width: 8, height: 8)
                                    Text(log.time).font(.caption).monospaced()
                                    Text(log.message).font(.subheadline)
                                    Spacer()
                                }
                                .padding(8)
                                .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.1)))
                            }
                        }
                        .padding()
                    }
                }
            }
        }
        // --- ADAPTIVE SAMPLING LOGIC ---
        .onReceive(power.$isOnBattery) { onBattery in
            // When power state changes, update the timer
            updateTimer()
        }
        .onReceive(timer) { _ in
            guard isMonitoring else { return }
            tick()
        }
    }
    
    // --- LOGIC ---
    
    func updateTimer() {
        // Ask the Warden for the correct speed
        let interval = power.recommendedInterval
        print("⚡️ Power State Changed. New Interval: \(interval)s")
        
        // Restart the timer with the new interval
        self.timer.upstream.connect().cancel()
        self.timer = Timer.publish(every: interval, on: .main, in: .common).autoconnect()
    }
    
    func tick() {
        // 1. Collect
        let data = eyes.getTelemetry()
        rawMetrics = data
        
        // 2. Analyze
        brain.analyze(data)
        
        // 3. Update History (Keep last 60 seconds)
        let now = Date()
        withAnimation {
            history.append(MetricPoint(time: now, score: brain.currentScore))
            if history.count > 60 { history.removeFirst() }
        }
        
        // 4. Log Anomalies
        if brain.isAnomalous {
            let msg = "Abnormal pattern detected (Score: \(String(format: "%.2f", brain.currentScore)))"
            // Simple de-bounce logic
            if logs.first?.type != .danger || logs.first?.time != DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium) {
                 addLog(msg, type: .danger)
            }
        }
    }
    
    func addLog(_ msg: String, type: LogEvent.EventType) {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        let log = LogEvent(time: formatter.string(from: Date()), message: msg, type: type)
        withAnimation {
            logs.insert(log, at: 0)
            if logs.count > 20 { logs.removeLast() }
        }
    }
    
    func formatBytes(_ bytes: Float) -> String {
        if bytes > 1024 * 1024 { return String(format: "%.1f MB/s", bytes / 1024 / 1024) }
        if bytes > 1024 { return String(format: "%.1f KB/s", bytes / 1024) }
        return String(format: "%.0f B/s", bytes)
    }
    
    func colorForEvent(_ type: LogEvent.EventType) -> Color {
        switch type {
        case .info: return .gray
        case .warning: return .orange
        case .danger: return .red
        }
    }
}

// --- SUBVIEWS ---

struct HeaderView: View {
    let score: Double
    let threshold: Double
    let isAnomalous: Bool
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Silicon Sentinel")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("Neural Engine Monitoring Active")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
            
            VStack(alignment: .trailing) {
                Text(isAnomalous ? "ANOMALY DETECTED" : "SYSTEM NORMAL")
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(6)
                    .background(isAnomalous ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                
                Text("MSE Score: \(String(format: "%.4f", score))")
                    .font(.monospacedDigit(.body)())
                    .foregroundColor(isAnomalous ? .red : .secondary)
            }
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                Text(title).font(.caption)
            }
            .foregroundColor(.secondary)
            
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .controlBackgroundColor)))
        .shadow(radius: 1)
    }
}

struct SidebarView: View {
    let status: String
    @Binding var monitoring: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Label("Dashboard", systemImage: "chart.xyaxis.line")
            Label("Settings", systemImage: "gear")
            
            Spacer()
            
            Divider()
            
            HStack {
                Circle()
                    .fill(monitoring ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(status)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Button(action: { monitoring.toggle() }) {
                Text(monitoring ? "Stop Monitoring" : "Start System")
                    .frame(maxWidth: .infinity)
            }
            .controlSize(.large)
            .tint(monitoring ? .red : .accentColor)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

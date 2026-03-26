//
//  ContentView.swift
//  CoreMetric
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

struct LogEvent: Identifiable, Codable {
    let id: UUID
    let time: String
    let message: String
    let type: EventType

    enum EventType: String, Codable {
        case info, warning, danger
    }

    init(id: UUID = UUID(), time: String, message: String, type: EventType) {
        self.id = id
        self.time = time
        self.message = message
        self.type = type
    }
}

// --- MAIN VIEW ---
struct ContentView: View {
    // Shared engine injected by MenuBarManager via .environmentObject
    @EnvironmentObject var brain: InferenceEngine
    @ObservedObject private var power = PowerManager.shared

    private let eyes = SystemCollector()

    // UI State
    @StateObject private var eventStore = EventStore()
    @StateObject private var orchestrator = SetupOrchestrator()
    @State private var history: [MetricPoint] = []
    @State private var isMonitoring = false
    @State private var rawMetrics: [Float] = Array(repeating: 0.0, count: 11)
    @State private var metricHistories: [[Float]] = Array(repeating: [], count: 11)
    @State private var lastDangerLogTime: Date = .distantPast
    @State private var selectedProcess: SuspectProcess? = nil
    @State private var showAllEvents = false
    @State private var showSetup = false
    @AppStorage("alertCooldown")       private var alertCooldown: Double  = 5.0
    @AppStorage("autoStartMonitoring") private var autoStart: Bool        = true

    // Timer — stored as AnyCancellable for reliable cancellation
    @State private var timerCancellable: AnyCancellable?

    var body: some View {
        NavigationSplitView {
            SidebarView(status: brain.modelStatus, monitoring: $isMonitoring)
        } detail: {
            VStack(spacing: 0) {
                // HEADER
                HeaderView(
                    score: brain.currentScore,
                    threshold: brain.alertThreshold,
                    isAnomalous: brain.isAnomalous,
                    isModelLoaded: brain.isModelLoaded,
                    thermalState: power.thermalState
                )
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))

                Divider()

                ScrollView {
                    VStack(spacing: 20) {

                        // --- CHART AREA ---
                        VStack(alignment: .leading) {
                            HStack {
                                Text("Real-time Anomaly Score")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                Spacer()
                                if power.isOnBattery {
                                    Label("Eco Mode", systemImage: "leaf.fill")
                                        .font(.caption)
                                        .foregroundColor(.green)
                                }
                            }

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
                            .chartYScale(domain: 0...max(brain.alertThreshold * 3, history.map(\.score).max() ?? 1.0))
                            .frame(height: 200)
                        }
                        .padding()
                        .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .controlBackgroundColor)))
                        .shadow(radius: 1)
                        .padding(.horizontal)
                        .padding(.top)

                        // --- DETECTIVE UI ---
                        if brain.isAnomalous && !brain.topSuspects.isEmpty {
                            VStack(alignment: .leading) {
                                Label("Potential Causes", systemImage: "exclamationmark.triangle.fill")
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
                                        Image(systemName: "chevron.right")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(6)
                                    .background(Color.red.opacity(0.08))
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                                    .contentShape(Rectangle())
                                    .onTapGesture { selectedProcess = process }
                                }
                            }
                            .padding()
                            .background(RoundedRectangle(cornerRadius: 8).stroke(Color.red, lineWidth: 1))
                            .padding(.horizontal)
                        }

                        // --- METRICS GRID ---
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 140))], spacing: 15) {
                            MetricCard(title: "CPU Load",     value: String(format: "%.1f%%", rawMetrics[0]),  icon: "cpu",                  history: metricHistories[0])
                            MetricCard(title: "Memory",       value: String(format: "%.1f%%", rawMetrics[1]),  icon: "memorychip",            history: metricHistories[1])
                            MetricCard(title: "Swap",         value: String(format: "%.1f%%", rawMetrics[2]),  icon: "arrow.2.circlepath",    history: metricHistories[2])
                            MetricCard(title: "System Load",  value: String(format: "%.2f",   rawMetrics[3]),  icon: "gauge",                 history: metricHistories[3])
                            MetricCard(title: "Network Up",   value: formatBytes(rawMetrics[4]),               icon: "arrow.up.circle",       history: metricHistories[4])
                            MetricCard(title: "Network Down", value: formatBytes(rawMetrics[5]),               icon: "arrow.down.circle",     history: metricHistories[5])
                            MetricCard(title: "Disk Read",    value: formatBytes(rawMetrics[6]),               icon: "arrow.down.doc",        history: metricHistories[6])
                            MetricCard(title: "Disk Write",   value: formatBytes(rawMetrics[7]),               icon: "arrow.up.doc",          history: metricHistories[7])
                            MetricCard(title: "Threads",      value: String(format: "%.0f",   rawMetrics[9]),  icon: "text.alignleft",        history: metricHistories[9])
                            MetricCard(title: "Processes",    value: String(format: "%.0f",   rawMetrics[10]), icon: "square.grid.2x2",       history: metricHistories[10])
                        }
                        .padding(.horizontal)

                        // --- LOGS ---
                        VStack(alignment: .leading) {
                            HStack {
                                Text("Recent Events")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                Spacer()
                                if !eventStore.events.isEmpty {
                                    Button("View All") { showAllEvents = true }
                                        .font(.caption)
                                        .buttonStyle(.plain)
                                        .foregroundColor(.accentColor)
                                }
                            }

                            if eventStore.events.isEmpty {
                                Text("No events yet.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding(.vertical, 8)
                            } else {
                                ForEach(eventStore.events.prefix(5)) { log in
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
                        }
                        .padding()
                    }
                }
                .overlay {
                    if !brain.isModelLoaded {
                        VStack(spacing: 12) {
                            Image(systemName: "sparkles")
                                .font(.largeTitle)
                                .foregroundColor(.accentColor)
                            Text("No Model Yet")
                                .font(.headline)
                            Text("CoreMetric needs to learn what's normal on your Mac before it can detect anomalies.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                            Button("Set Up Now") { showSetup = true }
                                .buttonStyle(.borderedProminent)
                                .padding(.top, 4)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color(nsColor: .windowBackgroundColor).opacity(0.92))
                    }
                }
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if !brain.isModelLoaded {
                    Button { showSetup = true } label: {
                        Label("Set Up", systemImage: "sparkles")
                    }
                    .help("Run setup to train a model")
                }
                Button {
                    isMonitoring.toggle()
                } label: {
                    Label(isMonitoring ? "Stop Monitoring" : "Start Monitoring",
                          systemImage: isMonitoring ? "stop.circle" : "play.circle")
                }
                .help(isMonitoring ? "Pause monitoring" : "Resume monitoring")
            }
        }
        .onAppear {
            isMonitoring = autoStart
            startTimer(interval: power.recommendedInterval)
        }
        .onReceive(power.$isOnBattery) { _ in
            startTimer(interval: power.recommendedInterval)
        }
        .onReceive(power.$thermalState) { _ in
            startTimer(interval: power.recommendedInterval)
        }
        .sheet(item: $selectedProcess) { process in
            ProcessDetailView(process: process)
        }
        .sheet(isPresented: $showAllEvents) {
            EventHistoryView(store: eventStore)
        }
        .sheet(isPresented: $showSetup) {
            SetupView(orchestrator: orchestrator) {
                brain.loadModel()
            }
        }
    }

    // --- LOGIC ---

    func startTimer(interval: TimeInterval) {
        timerCancellable?.cancel()
        timerCancellable = Timer.publish(every: interval, on: .main, in: .common)
            .autoconnect()
            .sink { [self] _ in
                guard isMonitoring else { return }
                tick()
            }
    }

    func tick() {
        // 1. Collect
        let data = eyes.getTelemetry()
        rawMetrics = data

        // 2. Update sparkline histories
        for i in 0..<min(data.count, metricHistories.count) {
            metricHistories[i].append(data[i])
            if metricHistories[i].count > 30 { metricHistories[i].removeFirst() }
        }

        // 3. Analyze — synchronous, so brain.currentScore is up to date immediately after
        brain.analyze(data)

        // 4. Update History (rolling 90-second window) — score is current now
        let now = Date()
        withAnimation {
            history.append(MetricPoint(time: now, score: brain.currentScore))
            let cutoff = now.addingTimeInterval(-90)
            history.removeAll { $0.time < cutoff }
        }

        // 5. Log Anomalies — debounced by alertCooldown setting
        if brain.isAnomalous {
            let msg = "Abnormal pattern detected (Score: \(String(format: "%.2f", brain.currentScore)))"
            if now.timeIntervalSince(lastDangerLogTime) > alertCooldown {
                lastDangerLogTime = now
                addLog(msg, type: .danger)
                NotificationManager.shared.sendAnomalyAlert(score: brain.currentScore)
            }
        }
    }

    func addLog(_ msg: String, type: LogEvent.EventType) {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        let log = LogEvent(time: formatter.string(from: Date()), message: msg, type: type)
        withAnimation { eventStore.add(log) }
    }

    func formatBytes(_ bytes: Float) -> String {
        if bytes > 1024 * 1024 { return String(format: "%.1f MB/s", bytes / 1024 / 1024) }
        if bytes > 1024 { return String(format: "%.1f KB/s", bytes / 1024) }
        return String(format: "%.0f B/s", bytes)
    }

    func colorForEvent(_ type: LogEvent.EventType) -> Color {
        switch type {
        case .info:    return .gray
        case .warning: return .orange
        case .danger:  return .red
        }
    }
}

// --- SUBVIEWS ---

struct HeaderView: View {
    let score: Double
    let threshold: Double
    let isAnomalous: Bool
    let isModelLoaded: Bool
    let thermalState: ProcessInfo.ThermalState

    private var thermalLabel: String {
        switch thermalState {
        case .critical: return "CRITICAL HEAT"
        case .serious:  return "Serious Heat"
        default:        return "Fair Heat"
        }
    }

    private var thermalColor: Color {
        thermalState == .critical ? .red : .orange
    }

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("CoreMetric")
                    .font(.title2)
                    .fontWeight(.bold)
                Text(isModelLoaded ? "Neural Engine Active" : "No model loaded")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text(isAnomalous ? "ANOMALY DETECTED" : "SYSTEM NORMAL")
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(6)
                    .background(isAnomalous ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                if thermalState != .nominal {
                    Label(thermalLabel, systemImage: "thermometer.high")
                        .font(.caption2)
                        .foregroundColor(thermalColor)
                }

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
    var history: [Float] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                Text(title).font(.caption)
            }
            .foregroundColor(.secondary)

            Text(value)
                .font(.title3)
                .fontWeight(.semibold)

            if !history.isEmpty {
                Chart(Array(history.enumerated()), id: \.offset) { i, v in
                    LineMark(
                        x: .value("t", i),
                        y: .value("v", Double(v))
                    )
                    .foregroundStyle(.blue.opacity(0.6))
                }
                .chartXAxis(.hidden)
                .chartYAxis(.hidden)
                .frame(height: 28)
            }
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
            Text("VIEWS")
                .font(.caption2)
                .foregroundColor(.secondary)
                .padding(.top, 4)
            Label("Dashboard", systemImage: "chart.xyaxis.line")
                .font(.subheadline)
                .foregroundColor(.primary)
            Button {
                NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
            } label: {
                Label("Settings", systemImage: "gear")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)

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

struct ProcessDetailView: View {
    let process: SuspectProcess
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "gearshape.2.fill")
                .font(.system(size: 40))
                .foregroundColor(.secondary)

            VStack(spacing: 4) {
                Text(process.name)
                    .font(.title2)
                    .fontWeight(.bold)
                Text("PID \(process.pid)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 24) {
                VStack {
                    Text(String(format: "%.1f%%", process.cpu))
                        .font(.title3).fontWeight(.semibold)
                    Text("CPU")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))

            HStack(spacing: 12) {
                Button("Open Activity Monitor") {
                    NSWorkspace.shared.open(
                        URL(fileURLWithPath: "/System/Applications/Utilities/Activity Monitor.app")
                    )
                    dismiss()
                }
                .buttonStyle(.borderedProminent)

                Button("Dismiss") { dismiss() }
                    .buttonStyle(.bordered)
            }
        }
        .padding(28)
        .frame(width: 300)
    }
}

struct EventHistoryView: View {
    @ObservedObject var store: EventStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Event History")
                    .font(.headline)
                Spacer()
                Button("Clear") { store.clear() }
                    .foregroundColor(.red)
                    .font(.caption)
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
            }
            .padding()

            Divider()

            if store.events.isEmpty {
                Spacer()
                Text("No events recorded.")
                    .foregroundColor(.secondary)
                Spacer()
            } else {
                List(store.events) { log in
                    HStack(alignment: .top, spacing: 10) {
                        Circle()
                            .fill(eventColor(log.type))
                            .frame(width: 8, height: 8)
                            .padding(.top, 4)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(log.message).font(.subheadline)
                            Text(log.time).font(.caption).foregroundColor(.secondary).monospaced()
                        }
                    }
                    .listRowBackground(Color.clear)
                }
                .listStyle(.plain)
            }
        }
        .frame(width: 420, height: 480)
    }

    private func eventColor(_ type: LogEvent.EventType) -> Color {
        switch type {
        case .info:    return .gray
        case .warning: return .orange
        case .danger:  return .red
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(InferenceEngine())
}
